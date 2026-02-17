import os, subprocess, logging, html, json, asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Setup ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # Global store: { pid: {proc, slug} }

logging.basicConfig(level=logging.INFO)

def esc(text):
    """Safely escape text for HTML parse mode."""
    return html.escape(str(text))

async def post_init(application):
    """Registers all 9 commands to the Telegram Menu."""
    commands = [
        BotCommand("start", "Introduction and help guide"),
        BotCommand("myfiles", "Explore files & actions"),
        BotCommand("upload", "Save & push file to GitHub"),
        BotCommand("run", "Start bot from bot.json"),
        BotCommand("stop", "Kill a running process"),
        BotCommand("logs", "View blockquote logs"),
        BotCommand("deployments", "List active tasks"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Permanently delete a file")
    ]
    await application.bot.set_my_commands(commands)
    logging.info("Commands successfully registered to Telegram menu.")

# --- UI Helper: HTML Blockquote Logs ---

async def get_logs_view(uid, slug):
    """Helper to fetch logs and wrap them in HTML blockquotes for the UI."""
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:] # Fetch last 15 lines
            log_content = "".join(lines).strip() if lines else "Empty log file."

    text = (f"📋 <b>Logs for:</b> <code>{esc(slug)}</code>\n\n"
            f"<b>Terminal Output:</b>\n"
            f"<blockquote><code>{esc(log_content)}</code></blockquote>")
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Logic: Dynamic Runner ---

async def dynamic_run(update, context, filename):
    """Logic to detect file type and execute with the correct runtime."""
    uid = update.effective_user.id
    user_path = engine.get_user_base(uid)
    ext = filename.split('.')[-1].lower()
    slug = filename.split('.')[0]
    pid = f"{uid}_{slug}"
    
    # Language mapping
    runtimes = {
        "py": f"{engine.get_venv_exe(uid)} {filename}",
        "js": f"node {filename}",
        "sh": f"bash {filename}"
    }
    cmd = runtimes.get(ext)
    if not cmd:
        return await update.effective_message.reply_text(f"❌ No runtime for <code>.{esc(ext)}</code>", parse_mode="HTML")

    log_p = os.path.join(user_path, f"{pid}.log")
    open(log_p, 'w').close() # Clear logs on start
    
    proc = subprocess.Popen(cmd, shell=True, cwd=user_path, stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
    else:
        await update.effective_message.reply_text(text, reply_markup=markup, parse_mode="HTML")

# --- Logic for the 9 Commands ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """1. Logic for /start: Explains all commands and shows navigation."""
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        "🤖 <b>Bot Lab Manager v42.0</b>\n"
        "Your environment is ready. Every upload is auto-synced to GitHub.\n\n"
        "📑 <b>COMMAND GUIDE</b>\n"
        "• /start - This introduction\n"
        "• /myfiles - Explore files & manage them\n"
        "• /upload [name] - Save a file & push to Git\n"
        "• /run - Execute your <code>bot.json</code>\n"
        "• /stop [slug] - Kill a running process\n"
        "• /logs [slug] - View logs in blockquotes\n"
        "• /deployments - List all active tasks\n"
        "• /send [slug] [txt] - Input to terminal\n"
        "• /delete [name] - Remove file permanently\n"
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def myfiles_cmd(update, context):
    """2. Logic for /myfiles: Scans directory and builds interactive file list."""
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
    """3. Logic for /upload: Handles file download and Git synchronization."""
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file with: <code>/upload name.py</code>", parse_mode="HTML")
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    
    success, err = engine.git_push_file(uid, fname)
    status = "✅ Saved & Pushed to GitHub" if success else f"⚠️ Error: {esc(err)}"
    await update.message.reply_text(status, parse_mode="HTML")

async def run_cmd(update, context):
    """LOGIC: Execute via bot.json OR run a direct shell command."""
    uid = update.effective_user.id
    
    # If the user typed something after /run (e.g., /run pip install telethon)
    if context.args:
        cmd = " ".join(context.args)
        # Force use of venv if they type 'pip' or 'python'
        venv_exe = engine.get_venv_exe(uid)
        cmd = cmd.replace("pip ", f"{venv_exe.replace('python3', 'pip')} ")
        cmd = cmd.replace("python ", f"{venv_exe} ")
        
        # Execute as a temporary task
        slug = "manual_cmd"
        await dynamic_run_raw(update, context, cmd, slug)
        return

    # Otherwise, fallback to bot.json logic
    config = engine.read_config(uid)
    if not config: 
        return await update.message.reply_text("❌ No command provided and <code>bot.json</code> missing.", parse_mode="HTML")
    
    main_file = config.get('main_file', 'bot.py')
    await dynamic_run(update, context, main_file)

async def dynamic_run_raw(update, context, cmd, slug):
    """Helper to run non-file shell commands."""
    uid = update.effective_user.id
    pid = f"{uid}_{slug}"
    log_p = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    
    proc = subprocess.Popen(cmd, shell=True, cwd=engine.get_user_base(uid), 
                            stdout=open(log_p, "w"), stderr=subprocess.STDOUT)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    await update.effective_message.reply_text(f"🛠 <b>Executing Task:</b> <code>{esc(cmd)}</code>\n\n{text}", 
                                             reply_markup=markup, parse_mode="HTML")

async def stop_cmd(update, context):
    """5. Logic for /stop: Terminates a specific process."""
    if not context.args and not update.callback_query: return
    slug = context.args[0] if context.args else update.callback_query.data.replace("stop_", "")
    pid = f"{update.effective_user.id}_{slug}"
    
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        msg = f"🛑 Stopped: <code>{esc(slug)}</code>"
    else: msg = "❌ Process not found."
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, parse_mode="HTML")

async def logs_cmd(update, context):
    """6. Logic for /logs: Manual log retrieval via command."""
    if not context.args: return await update.message.reply_text("❌ Usage: /logs [slug]")
    text, markup = await get_logs_view(update.effective_user.id, context.args[0])
    await update.effective_message.reply_text(text, reply_markup=markup, parse_mode="HTML")

async def deployments_cmd(update, context):
    """7. Logic for /deployments: Lists all active bots for the user."""
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 <b>Active Bots:</b>\n" + "\n".join([f"✅ <code>{esc(p)}</code>" for p in active]) if active else "📭 No active tasks."
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")

async def send_cmd(update, context):
    """8. Logic for /send: Sends input to a running terminal."""
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: /send [slug] [text]")
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Input sent to process.")
    else:
        await update.message.reply_text("❌ Process not found.")

async def delete_cmd(update, context):
    """9. Logic for /delete: Removes a file from the server."""
    if not context.args: return
    fname = context.args[0]
    path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path):
        os.remove(path)
        msg = f"🗑 Deleted: <code>{esc(fname)}</code>"
    else: msg = "❌ File not found."
    await update.effective_message.reply_text(msg, parse_mode="HTML")

# --- Interactive Callback Handler ---

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles": await myfiles_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data.startswith("fopt_"):
        fname = data.replace("fopt_", "")
        row1 = [InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{fname}")]
        if fname == "requirements.txt": row1.append(InlineKeyboardButton("📦 Install Deps", callback_data="inst_deps"))
        kb = [row1, 
              [InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}"), 
               InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 <b>File:</b> <code>{esc(fname)}</code>", reply_markup=InlineKeyboardMarkup(kb), parse_mode="HTML")
    elif data.startswith("qrun_"): await dynamic_run(update, context, data.replace("qrun_", ""))
    elif data == "inst_deps":
        pip_exe = engine.get_venv_exe(uid).replace("python3", "pip")
        subprocess.Popen(f"{pip_exe} install -r requirements.txt", shell=True, cwd=engine.get_user_base(uid))
        await query.edit_message_text("⏳ <b>Dependencies installing in venv...</b>", parse_mode="HTML")
    elif data.startswith("logref_"):
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="HTML")
        except BadRequest: pass
    elif data.startswith("stop_"): await stop_cmd(update, context)
    elif data.startswith("fdel_"):
        fname = data.replace("fdel_", ""); context.args = [fname]; await delete_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # 9 Full Logic Handlers
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
