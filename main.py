import os, subprocess, logging, re, asyncio, shutil, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from engine import LabEngine

# --- Initialization ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # Stores { pid: {proc, slug, uid} }

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def escape_md(text):
    """Escapes special characters for Telegram MarkdownV2."""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

async def start_configured_process(uid, config, context):
    """Starts the sub-bot using the user's specific virtual environment."""
    slug = config['name']
    pid = f"{uid}_{slug}"
    exe = engine.get_venv_exe(uid)
    user_path = engine.get_user_base(uid)
    log_p = os.path.join(user_path, f"{pid}.log")

    # Replace 'python3' with the absolute path to the venv python
    cmd = config['start_cmd'].replace("python3", exe)
    
    # Start the process in unbuffered mode so logs appear instantly
    proc = subprocess.Popen(
        cmd, shell=True, cwd=user_path, 
        env={"PYTHONUNBUFFERED":"1", **config.get("env", {})}, 
        stdout=open(log_p, "w"), stderr=subprocess.STDOUT, 
        stdin=subprocess.PIPE, text=True, bufsize=0
    )

    running_processes[pid] = {"proc": proc, "slug": slug, "uid": uid}
    await context.bot.send_message(uid, f"🚀 *Started:* `{escape_md(slug)}`", parse_mode="MarkdownV2")

# --- Command Handlers ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Introduces the bot and lists all available commands."""
    uid = update.effective_user.id
    engine.setup_venv(uid) # Ensure user folder and venv exist
    
    msg = (
        r"🤖 *Bot Lab Manager v22\.0*" + "\n"
        r"⚡ *BRANCH-SYNC EDITION*" + "\n\n"
        r"📂 *FILE MANAGEMENT*" + "\n"
        r"• `/upload [name]` — Save file \& push to GitHub branch\." + "\n"
        r"• `/status` — View local files \& delete them\." + "\n"
        r"• `/delete [name]` — Remove a file locally\." + "\n\n"
        r"▶️ *EXECUTION*" + "\n"
        r"• `/run` — Start bot based on `bot.json`\." + "\n"
        r"• `/deployments` — List your running tasks\." + "\n"
        r"• `/stop [slug]` — Kill a running process\." + "\n"
        r"• `/logs [slug]` — See the last 15 lines of output\."
    )
    kb = [[
        InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
        InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")
    ]]
    
    # FIX: Check if triggered by button or /command to prevent AttributeError
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    """Saves a file locally and automatically pushes it to the user's GitHub branch."""
    if not update.message.reply_to_message or not context.args: 
        return await update.message.reply_text("❌ Usage: Reply to a file/text with `/upload filename.py` ")
    
    uid = update.effective_user.id
    filename = context.args[0]
    target = os.path.join(engine.get_user_base(uid), filename)
    
    # 1. Capture Content
    replied = update.message.reply_to_message
    if replied.document:
        f = await (await replied.document.get_file()).download_as_bytearray()
        with open(target, "wb") as file: file.write(f)
    else:
        with open(target, "w") as file: file.write(replied.text.strip())
    
    # 2. Automated GitHub Push to User Branch
    success, git_msg = engine.git_push_file(uid, filename)
    
    if success:
        await update.message.reply_text(f"✅ Saved \& Pushed to branch `user_{uid}`", parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(f"⚠️ Saved locally, but GitHub Push failed: `{escape_md(git_msg)}`", parse_mode="MarkdownV2")

async def status_cmd(update, context):
    """Shows local files and provides management buttons."""
    uid, base = update.effective_user.id, engine.get_user_base(update.effective_user.id)
    files = [f for f in os.listdir(base) if f != "venv" and not f.endswith(".log") and not f.startswith(".git")]
    
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f[:50]}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    
    if update.callback_query:
        await update.callback_query.edit_message_text("📂 *Local Files:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text("📂 *Local Files:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def run_cmd(update, context):
    """Executes the bot startup based on bot.json config."""
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config:
        return await update.message.reply_text("❌ `bot.json` not found. Upload it first!")
    await start_configured_process(uid, config, context)

async def stop_cmd(update, context):
    """Terminates a running process."""
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{context.args[0]}`")

async def cb_handler(update, context):
    """Handles all inline button clicks."""
    query = update.callback_query
    data = query.data
    await query.answer()
    
    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys":
        uid_prefix = f"{query.from_user.id}_"
        active = [v['slug'] for k, v in running_processes.items() if k.startswith(uid_prefix)]
        msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else "📭 No active tasks."
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="nav_home")]]), parse_mode="MarkdownV2")

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
