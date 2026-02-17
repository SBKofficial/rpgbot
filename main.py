import os, subprocess, logging, re, asyncio, json, shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Initialization ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # Stores { pid: {proc, slug} }

logging.basicConfig(level=logging.INFO)

def escape_md(text):
    """Escapes all reserved characters for Telegram MarkdownV2."""
    # The dot '.' was causing your crash; this regex handles it and others.
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# --- Command Registration (Auto-Upload to Telegram) ---
async def post_init(application):
    """Registers the complete 9-command list to the Telegram Menu button."""
    commands = [
        BotCommand("start", "Introduction and full guide"),
        BotCommand("myfiles", "File explorer & quick actions"),
        BotCommand("upload", "Save file & push to GitHub"),
        BotCommand("run", "Start bot from bot.json"),
        BotCommand("stop", "Kill a running process [slug]"),
        BotCommand("logs", "View blockquote logs [slug]"),
        BotCommand("deployments", "List all active bots"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Permanently delete a file [name]")
    ]
    await application.bot.set_my_commands(commands)
    logging.info("Commands successfully registered to Telegram menu.")

# --- UI Helpers ---
async def get_logs_view(uid, slug):
    """Generates the log view with blockquotes and control buttons."""
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Empty log file."

    # Format into blockquotes: > followed by each line
    formatted_logs = "\n".join([f">{line}" for line in log_content.split("\n")])
    text = f"📋 *Logs for:* `{escape_md(slug)}`\n\n{escape_md(formatted_logs)}"
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Command Handlers ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v33\.0*" + "\n"
        r"Every `/upload` auto\-syncs to your GitHub branch\." + "\n\n"
        r"📑 *COMPLETE COMMAND LIST*" + "\n"
        r"• `/start` — Show this manual\." + "\n"
        r"• `/myfiles` — Explore files with Run/Delete buttons\." + "\n"
        r"• `/upload [name]` — Save \& push file to GitHub\." + "\n"
        r"• `/run` — Execute bot based on `bot.json`\." + "\n"
        r"• `/stop [slug]` — Stop a running bot process\." + "\n"
        r"• `/logs [slug]` — Blockquote logs with Refresh button\." + "\n"
        r"• `/deployments` — List all active tasks\." + "\n"
        r"• `/send [slug] [txt]` — Send terminal input\." + "\n"
        r"• `/delete [name]` — Permanently delete a file\."
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        else:
            await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    except BadRequest: pass

async def myfiles_cmd(update, context):
    uid = update.effective_user.id
    files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
    
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    
    text = "📂 *Your Lab Files:*"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Usage: Reply to a file with `/upload filename.py` ")
    
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    
    # Auto-Push to GitHub (Engine logic)
    success, err = engine.git_push_file(uid, fname)
    status = "✅ Saved & Pushed to GitHub" if success else f"⚠️ Saved Locally (Git Error: {escape_md(err)})"
    await update.message.reply_text(status, parse_mode="MarkdownV2")

async def run_cmd(update, context):
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config: return await update.message.reply_text("❌ No `bot.json` found.")
    
    slug = config['name']; pid = f"{uid}_{slug}"
    log_p = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    if os.path.exists(log_p): open(log_p, 'w').close()
    
    proc = subprocess.Popen(config['start_cmd'].replace("python3", engine.get_venv_exe(uid)), 
                            shell=True, cwd=engine.get_user_base(uid), stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    await update.effective_message.reply_text(f"🚀 *Started\!*\n\n{text}", reply_markup=markup, parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/logs slug` ")
    text, markup = await get_logs_view(update.effective_user.id, context.args[0])
    await update.message.reply_text(text, reply_markup=markup, parse_mode="MarkdownV2")

async def stop_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/stop slug` ")
    slug = context.args[0]; pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate(); del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{escape_md(slug)}`", parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 *Active Bots:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else "📭 No active tasks."
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/delete filename` ")
    fname = context.args[0]; path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path):
        os.remove(path)
        await update.message.reply_text(f"🗑 Deleted `{escape_md(fname)}`", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: `/send slug text` ")
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to terminal.")

# --- Button Callback Handler ---

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    
    elif data.startswith("fopt_"):
        fname = data.replace("fopt_", "")
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{fname}"),
               InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}")],
              [InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}"),
               InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(fname)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("logref_"):
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="MarkdownV2")
        except BadRequest: pass

    elif data.startswith("stop_"):
        slug = data.replace("stop_", ""); context.args = [slug]; await stop_cmd(update, context)

    elif data.startswith("fdel_"):
        fname = data.replace("fdel_", ""); context.args = [fname]; await delete_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # Registering all Command Handlers to match the registered menu
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("myfiles", myfiles_cmd))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CommandHandler("deployments", deployments_cmd))
    app.add_handler(CommandHandler("send", send_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
