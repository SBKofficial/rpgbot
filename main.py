import os, subprocess, logging, re, asyncio, json, shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from engine import LabEngine

# --- Initialization ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, slug, uid} }

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def escape_md(text):
    """Escapes special characters for Telegram MarkdownV2."""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# --- Command Handlers ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Acts as an introduction to the bot and explains all available commands.
    """
    uid = update.effective_user.id
    engine.setup_venv(uid)
    
    msg = (
        r"🤖 *Welcome to Bot Lab Manager v25\.0*" + "\n"
        r"This bot allows you to host, run, and sync your Python scripts directly to GitHub branches\." + "\n\n"
        
        r"📑 *COMMAND GUIDE*" + "\n\n"
        
        r"📂 *File Management*" + "\n"
        r"• `/upload [name]` — Save a file locally \& auto\-push to your GitHub branch\. Reply to a file or code block with this\." + "\n"
        r"• `/myfiles` — View all files in your lab and manage them\." + "\n"
        r"• `/delete [name]` — Permanently remove a file from your lab\." + "\n\n"
        
        r"🚀 *Execution*" + "\n"
        r"• `/run` — Starts your bot using the settings in `bot.json`\." + "\n"
        r"• `/stop [slug]` — Force stop a running process\." + "\n"
        r"• `/logs [slug]` — See the most recent output from your bot\." + "\n"
        r"• `/send [slug] [text]` — Send text input to a running bot's terminal\." + "\n"
        r"• `/deployments` — See all your currently active bots\." + "\n\n"
        
        r"💡 *Note:* Every time you `/upload`, your changes are backed up to the `user_" + str(uid) + r"` branch on your repository\."
    )
    
    kb = [
        [InlineKeyboardButton("📂 My Files", callback_data="files_refresh"),
         InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")],
        [InlineKeyboardButton("❓ Help / Support", url="https://t.me/your_support_link")]
    ]
    
    # Use effective_message to handle both /start and Home button clicks
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    if not update.message.reply_to_message or not context.args: 
        return await update.message.reply_text("❌ Usage: Reply to a file with `/upload filename.py` ")
    
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    
    # Auto-push to GitHub branch via engine
    success, err = engine.git_push_file(uid, fname)
    if success:
        await update.message.reply_text(f"✅ Saved \& Pushed to branch `user_{uid}`", parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(f"⚠️ Saved locally, but GitHub push failed: `{escape_md(err)}`", parse_mode="MarkdownV2")

async def myfiles_cmd(update, context):
    uid = update.effective_user.id
    base = engine.get_user_base(uid)
    files = [f for f in os.listdir(base) if f not in ["venv", ".git"] and not f.endswith(".log")]
    
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f[:50]}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    
    text = "📂 *Your Lab Files:*"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def run_cmd(update, context):
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config: return await update.message.reply_text("❌ `bot.json` missing! Upload it to start.")
    
    pid = f"{uid}_{config['name']}"
    log_p = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    
    proc = subprocess.Popen(
        config['start_cmd'].replace("python3", engine.get_venv_exe(uid)), 
        shell=True, cwd=engine.get_user_base(uid), 
        stdout=open(log_p, "w"), stderr=subprocess.STDOUT, 
        stdin=subprocess.PIPE, text=True, bufsize=0
    )
    
    running_processes[pid] = {"proc": proc, "slug": config['name']}
    await update.message.reply_text(f"🚀 Started: `{escape_md(config['name'])}`", parse_mode="MarkdownV2")

async def stop_cmd(update, context):
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{context.args[0]}`")

async def deployments_cmd(update, context):
    uid_prefix = f"{update.effective_user.id}_"
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(uid_prefix)]
    msg = "🛰 *Active Deployments:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else "📭 No active tasks."
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(msg, parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            await update.message.reply_text(f"📋 *Logs for {slug}:*\n`{''.join(lines)}`", parse_mode="MarkdownV2")
    else:
        await update.message.reply_text("❌ No logs found.")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Input sent to terminal.")

async def delete_cmd(update, context):
    if not context.args: return
    fname = context.args[0]
    path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path):
        if os.path.isdir(path): shutil.rmtree(path)
        else: os.remove(path)
        await update.message.reply_text(f"🗑 Deleted `{fname}`")

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; await query.answer()
    if data == "nav_home": await start_cmd(update, context)
    elif data == "files_refresh": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("myfiles", myfiles_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("deployments", deployments_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("send", send_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
