import os, subprocess, logging, html, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Initialization ---
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
        BotCommand("run", "Start bot from bot.json or command"),
        BotCommand("stop", "Kill a running process"),
        BotCommand("logs", "View blockquote logs"),
        BotCommand("deployments", "List active tasks"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Permanently delete a file")
    ]
    await application.bot.set_my_commands(commands)

# --- UI Helper: HTML Blockquote Logs ---

async def get_logs_view(uid, slug):
    """Logic to fetch terminal output and wrap in HTML blockquotes."""
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Empty log file."

    text = (f"📋 <b>Logs for:</b> <code>{esc(slug)}</code>\n\n"
            f"<blockquote><code>{esc(log_content)}</code></blockquote>\n"
            f"<i>(Click Refresh to see new updates)</i>")
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Core Execution Logic ---

async def execute_shell(update, context, cmd, slug):
    """Handles subprocess creation for Python, Node, Pip, or Npm."""
    uid = update.effective_user.id
    user_path = engine.get_user_base(uid)
    pid = f"{uid}_{slug}"
    log_p = os.path.join(user_path, f"{pid}.log")
    
    # Reset log and start process
    open(log_p, 'w').close()
    proc = subprocess.Popen(cmd, shell=True, cwd=user_path, stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    msg = f"🚀 <b>Running:</b> <code>{esc(cmd)}</code>\n\n{text}"
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=markup, parse_mode="HTML")

# --- Function Logics for all 9 Commands ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    1. Intro & Command Guide Logic.
    Now includes GitHub recovery to prevent 'forgotten files' on restart.
    """
    uid = update.effective_user.id
    
    # Send a status update so the user knows recovery is happening
    status_msg = "🛰 <b>Initializing Lab...</b>\nConnecting to GitHub to recover your files."
    if update.callback_query:
        await update.callback_query.edit_message_text(status_msg, parse_mode="HTML")
        sent_msg = update.callback_query.message
    else:
        sent_msg = await update.effective_message.reply_text(status_msg, parse_mode="HTML")

    # --- RECOVERY LOGIC ---
    # This pulls your files back from the cloud before showing the menu
    success, git_status = engine.sync_from_github(uid)
    engine.setup_venv(uid)
    # ----------------------

    msg = (
        "🤖 <b>Bot Lab Manager v52.0</b>\n"
        f"📡 Status: <i>{esc(git_status)}</i>\n\n"
        "Welcome! I am your persistent cloud terminal. Use the commands below to manage your projects:\n\n"
        "📑 <b>COMMAND MANUAL</b>\n"
        "• /start - Reset and view this manual\n"
        "• /myfiles - View your recovered files and manage them\n"
        "• /upload [name] - Save a file and <b>Sync to GitHub</b>\n"
        "• /run [cmd] - Execute <code>npm</code>, <code>pip</code>, or <code>bot.json</code>\n"
        "• /stop [slug] - Kill a running process immediately\n"
        "• /logs [slug] - View real-time terminal output\n"
        "• /deployments - List all active background tasks\n"
        "• /send [slug] [txt] - Send input to a process (phone/code)\n"
        "• /delete [name] - Remove file from local and cloud\n\n"
        "<i>All work is automatically saved to your GitHub branch.</i>"
    )
    
    kb = [
        [InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
         InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")],
        [InlineKeyboardButton("🏠 Refresh Home", callback_data="nav_home")]
    ]

    await sent_msg.edit_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def run_cmd(update, context):
    """
    2. Runner logic supporting direct shell or bot.json.
    Automatically routes 'pip' to the local Venv.
    """
    uid = update.effective_user.id
    if not context.args:
        # If no args, look for a bot.json configuration
        config = engine.read_config(uid)
        if config:
            cmd = f"{engine.get_venv_exe(uid)} {config.get('main_file', 'bot.py')}"
            return await execute_shell(update, context, cmd, config.get('name', 'bot'))
        return await update.effective_message.reply_text("❌ <b>Usage:</b>\n<code>/run pip install [pkg]</code>\n<code>/run node [file].js</code>", parse_mode="HTML")

    raw_cmd = " ".join(context.args)
    
    # Auto-Venv routing for pip
    if raw_cmd.startswith("pip "):
        venv_pip = engine.get_venv_exe(uid).replace("python3", "pip")
        raw_cmd = raw_cmd.replace("pip ", f"{venv_pip} ", 1)

    await execute_shell(update, context, raw_cmd, "terminal")

async def myfiles_cmd(update, context):
    """3. File Explorer Logic."""
    uid = update.effective_user.id
    files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
    kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    
    if update.callback_query:
        await update.callback_query.edit_message_text("📂 <b>Files:</b>", reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text("📂 <b>Files:</b>", reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def upload_cmd(update, context):
    """4. Upload logic with immediate GitHub Sync."""
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file with: <code>/upload name.py</code>", parse_mode="HTML")
    
    uid, fname = update.effective_user.id, context.args[0]
    user_path = engine.get_user_base(uid)
    local_path = os.path.join(user_path, fname)
    
    # 1. Download from Telegram
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    
    with open(local_path, "wb") as f: 
        f.write(content)
    
    # 2. Sync to GitHub immediately
    success, git_log = engine.git_push_file(uid, fname)
    
    if success:
        await update.message.reply_text(
            f"✅ <b>Saved & Synced!</b>\n"
            f"📄 File: <code>{esc(fname)}</code>\n"
            f"🛰 Status: <code>{esc(git_log)}</code>", 
            parse_mode="HTML"
        )
    else:
        await update.message.reply_text(
            f"⚠️ <b>Saved locally, but Git Sync failed:</b>\n"
            f"<blockquote><code>{esc(git_log)}</code></blockquote>", 
            parse_mode="HTML"
        )


async def stop_cmd(update, context):
    """5. Kill process logic."""
    slug = context.args[0] if context.args else update.callback_query.data.replace("stop_", "") if update.callback_query else None
    if not slug: return
    pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate(); del running_processes[pid]
        msg = f"🛑 Stopped: <code>{esc(slug)}</code>"
    else: msg = "❌ Process not found."
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, parse_mode="HTML")

async def logs_cmd(update, context):
    """6. Log viewer logic."""
    if not context.args: return
    text, markup = await get_logs_view(update.effective_user.id, context.args[0])
    await update.effective_message.reply_text(text, reply_markup=markup, parse_mode="HTML")

async def deployments_cmd(update, context):
    """7. Deployment status logic."""
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 <b>Active Bots:</b>\n" + "\n".join([f"✅ <code>{esc(p)}</code>" for p in active]) if active else "📭 No active tasks."
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def send_cmd(update, context):
    """8. STDIN input logic with newline."""
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to terminal.")

async def delete_cmd(update, context):
    """9. File deletion logic."""
    fname = context.args[0] if context.args else update.callback_query.data.replace("fdel_", "") if update.callback_query else None
    if not fname: return
    path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path): os.remove(path)
    msg = f"🗑 Deleted: <code>{esc(fname)}</code>"
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, parse_mode="HTML")

# --- All Button Callback Logics ---

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    
    elif data.startswith("fopt_"): # CLICKED A FILE NAME
        fname = data.replace("fopt_", "")
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{fname}")],
              [InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}"), 
               InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 <b>File:</b> <code>{esc(fname)}</code>", reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

    elif data.startswith("qrun_"): # RUN BUTTON
        f = data.replace("qrun_", ""); cmd = f"node {f}" if f.endswith(".js") else f"{engine.get_venv_exe(uid)} {f}"
        await execute_shell(update, context, cmd, f.split('.')[0])

    elif data.startswith("logref_"): # REFRESH LOGS
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except BadRequest: pass

    elif data.startswith("stop_"): await stop_cmd(update, context)
    elif data.startswith("fdel_"): await delete_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # Mapping all handlers
    handlers = [
        CommandHandler("start", start_cmd), CommandHandler("myfiles", myfiles_cmd),
        CommandHandler("upload", upload_cmd), CommandHandler("run", run_cmd),
        CommandHandler("stop", stop_cmd), CommandHandler("logs", logs_cmd),
        CommandHandler("deployments", deployments_cmd), CommandHandler("send", send_cmd),
        CommandHandler("delete", delete_cmd), CallbackQueryHandler(cb_handler)
    ]
    for h in handlers: app.add_handler(h)
    
    app.run_polling()
