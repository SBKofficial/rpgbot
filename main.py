import os, subprocess, logging, html, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Setup ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, slug} }

logging.basicConfig(level=logging.INFO)

def esc(text):
    """Safely escape text for HTML parse mode."""
    return html.escape(str(text))

async def post_init(application):
    """Registers all 9 commands to the Telegram Menu button."""
    commands = [
        BotCommand("start", "Manual and Introduction"),
        BotCommand("myfiles", "File explorer & actions"),
        BotCommand("upload", "Save & push file to GitHub"),
        BotCommand("run", "Start bot from bot.json"),
        BotCommand("stop", "Kill a running process"),
        BotCommand("logs", "View blockquote logs"),
        BotCommand("deployments", "List active tasks"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Permanently delete a file")
    ]
    await application.bot.set_my_commands(commands)

# --- UI Helper: HTML Blockquote Logs ---

async def get_logs_view(uid, slug):
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Empty log file."

    # Logic for HTML formatting
    text = (f"📋 <b>Logs for:</b> <code>{esc(slug)}</code>\n\n"
            f"<blockquote><code>{esc(log_content)}</code></blockquote>")
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Command Logics (Individual Functions) ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """LOGIC: Setup environment and show the main menu with the command guide."""
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        "🤖 <b>Bot Lab Manager v40.0</b>\n"
        "Your workspace is initialized.\n\n"
        "📑 <b>COMMAND GUIDE</b>\n"
        "• /start - This manual\n"
        "• /myfiles - File list with Run buttons\n"
        "• /upload - Reply to a file to Save & Push\n"
        "• /run - Run based on <code>bot.json</code>\n"
        "• /stop [slug] - Kill a process\n"
        "• /logs [slug] - Show terminal logs\n"
        "• /deployments - List all active bots\n"
        "• /send [slug] [text] - Terminal input\n"
        "• /delete [filename] - Delete file\n"
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    target = update.callback_query.edit_message_text if update.callback_query else update.effective_message.reply_text
    await target(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def myfiles_cmd(update, context):
    """LOGIC: Scan directory and build a button list of user files."""
    uid = update.effective_user.id
    files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    await update.effective_message.reply_text("📂 <b>Your Lab Files:</b>", reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def upload_cmd(update, context):
    """LOGIC: Download replied file and auto-push to GitHub via Engine."""
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file with: /upload name.py")
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    
    success, err = engine.git_push_file(uid, fname)
    status = "✅ Saved & Pushed" if success else f"⚠️ Local Only (Error: {esc(err)})"
    await update.message.reply_text(status, parse_mode="HTML")

async def run_cmd(update, context):
    """LOGIC: Execute file using bot.json configuration."""
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config: return await update.message.reply_text("❌ <code>bot.json</code> not found.")
    
    slug = config.get('name', 'bot')
    # Dynamic logic: Replace python3 with the venv path
    cmd = config['start_cmd'].replace("python3", engine.get_venv_exe(uid))
    
    pid = f"{uid}_{slug}"
    log_p = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    open(log_p, 'w').close()
    
    proc = subprocess.Popen(cmd, shell=True, cwd=engine.get_user_base(uid), stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    await update.effective_message.reply_text(f"🚀 <b>Started:</b> {esc(slug)}\n\n{text}", reply_markup=markup, parse_mode="HTML")

async def stop_cmd(update, context):
    """LOGIC: Terminate a subprocess by slug name."""
    if not context.args: return await update.message.reply_text("❌ Usage: /stop slug")
    slug = context.args[0]; pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped: <code>{esc(slug)}</code>", parse_mode="HTML")
    else:
        await update.message.reply_text("❌ Process not found.")

async def logs_cmd(update, context):
    """LOGIC: Fetch the last lines of the log file and wrap in HTML blockquotes."""
    if not context.args: return await update.message.reply_text("❌ Usage: /logs slug")
    text, markup = await get_logs_view(update.effective_user.id, context.args[0])
    await update.message.reply_text(text, reply_markup=markup, parse_mode="HTML")

async def deployments_cmd(update, context):
    """LOGIC: Filter global process list for current user's active bots."""
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 <b>Active Bots:</b>\n" + "\n".join([f"✅ <code>{esc(p)}</code>" for p in active]) if active else "📭 No active tasks."
    await update.effective_message.reply_text(msg, parse_mode="HTML")

async def send_cmd(update, context):
    """LOGIC: Write text to a running process's STDIN."""
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: /send slug message")
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to terminal.")

async def delete_cmd(update, context):
    """LOGIC: Delete file from local workspace."""
    if not context.args: return await update.message.reply_text("❌ Usage: /delete filename")
    fname = context.args[0]; path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path):
        os.remove(path)
        await update.message.reply_text(f"🗑 Deleted <code>{esc(fname)}</code>", parse_mode="HTML")

# --- Callback/Button Logic ---

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data.startswith("logref_"):
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except BadRequest: pass

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # 9 Distinct Logic Mappings
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
