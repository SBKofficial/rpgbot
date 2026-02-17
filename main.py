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
    """Safely escape text for HTML parse mode to prevent 'Bad Request' crashes."""
    return html.escape(str(text))

async def post_init(application):
    """Registers all 9 commands to the Telegram Menu."""
    commands = [
        BotCommand("start", "Introduction and help guide"),
        BotCommand("myfiles", "Explore files & actions"),
        BotCommand("upload", "Save & push file to GitHub"),
        BotCommand("run", "Run command or bot.json"),
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

    # Using <blockquote> and HTML tags to prevent reserved character errors
    text = (f"📋 <b>Logs for:</b> <code>{esc(slug)}</code>\n\n"
            f"<blockquote><code>{esc(log_content)}</code></blockquote>")
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Logic: Universal Command Runner ---

async def execute_shell(update, context, cmd, slug):
    """Executes any shell command (pip, npm, bash) and redirects to logs."""
    uid = update.effective_user.id
    user_path = engine.get_user_base(uid)
    pid = f"{uid}_{slug}"
    
    log_p = os.path.join(user_path, f"{pid}.log")
    open(log_p, 'w').close()
    
    # Start process with redirected output
    proc = subprocess.Popen(cmd, shell=True, cwd=user_path, stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    msg = f"🛠 <b>Executing:</b> <code>{esc(cmd)}</code>\n\n{text}"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=markup, parse_mode="HTML")

# --- All 9 Command Logics ---

async def start_cmd(update, context):
    """1. Logic for /start: Introduction and Command Manual."""
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        "🤖 <b>Bot Lab Manager v43.0</b>\n"
        "Your environment is ready. HTML mode is active to prevent crashes.\n\n"
        "📑 <b>COMMAND GUIDE</b>\n"
        "• /start - This guide\n"
        "• /myfiles - File list & interactive buttons\n"
        "• /upload [name] - Reply to file to save & push\n"
        "• /run [cmd] - Run <code>npm</code>, <code>pip</code>, or <code>bot.json</code>\n"
        "• /stop [slug] - Kill process\n"
        "• /logs [slug] - View output\n"
        "• /deployments - Active tasks\n"
        "• /send [slug] [txt] - Terminal input\n"
        "• /delete [name] - Remove file\n"
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def run_cmd(update, context):
    """2. Logic for /run: Handles bot.json OR direct npm/pip/shell commands."""
    uid = update.effective_user.id
    if not context.args:
        # Check for bot.json if no args provided
        config = engine.read_config(uid)
        if config:
            main_file = config.get('main_file', 'bot.py')
            cmd = f"{engine.get_venv_exe(uid)} {main_file}"
            await execute_shell(update, context, cmd, config.get('name', 'bot'))
        else:
            await update.effective_message.reply_text("❌ Provide a command (e.g., <code>/run npm install</code>) or upload <code>bot.json</code>.", parse_mode="HTML")
        return

    raw_cmd = " ".join(context.args)
    # Auto-fix pathing for common managers
    if raw_cmd.startswith("pip "):
        venv_pip = engine.get_venv_exe(uid).replace("python3", "pip")
        raw_cmd = raw_cmd.replace("pip ", f"{venv_pip} ", 1)
    
    await execute_shell(update, context, raw_cmd, "terminal")

async def myfiles_cmd(update, context):
    """3. Logic for /myfiles: File Explorer."""
    uid = update.effective_user.id
    files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    msg = "📂 <b>Your Lab Files:</b>"
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def upload_cmd(update, context):
    """4. Logic for /upload: Git Push synchronization."""
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file with: <code>/upload name.py</code>", parse_mode="HTML")
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    success, err = engine.git_push_file(uid, fname)
    await update.message.reply_text("✅ Saved & Pushed" if success else f"⚠️ Git Error: {esc(err)}", parse_mode="HTML")

async def stop_cmd(update, context):
    """5. Logic for /stop: Process termination."""
    slug = context.args[0] if context.args else update.callback_query.data.replace("stop_", "") if update.callback_query else None
    if not slug: return
    pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        msg = f"🛑 Stopped: <code>{esc(slug)}</code>"
    else:
        msg = "❌ Process not found."
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, parse_mode="HTML")

async def logs_cmd(update, context):
    """6. Logic for /logs: Fetch current output."""
    if not context.args: return
    text, markup = await get_logs_view(update.effective_user.id, context.args[0])
    await update.effective_message.reply_text(text, reply_markup=markup, parse_mode="HTML")

async def deployments_cmd(update, context):
    """7. Logic for /deployments: List active tasks (Fixed HTML)."""
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 <b>Active Bots:</b>\n" + "\n".join([f"✅ <code>{esc(p)}</code>" for p in active]) if active else "📭 No active tasks."
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def send_cmd(update, context):
    """8. Logic for /send: Input to process STDIN."""
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to terminal.")

async def delete_cmd(update, context):
    """9. Logic for /delete: Remove local file."""
    if not context.args: return
    path = os.path.join(engine.get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(path):
        os.remove(path)
        await update.message.reply_text(f"🗑 Deleted: {esc(context.args[0])}", parse_mode="HTML")

# --- Callback & Polling ---

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; await query.answer()
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data.startswith("logref_"):
        text, markup = await get_logs_view(query.from_user.id, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except BadRequest: pass
    elif data.startswith("stop_"): await stop_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
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
