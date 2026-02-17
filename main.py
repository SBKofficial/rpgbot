import os, subprocess, logging, re, asyncio, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Setup ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, slug} }

logging.basicConfig(level=logging.INFO)

def escape_md(text):
    """Strictly escapes reserved characters for MarkdownV2 to prevent crashes."""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

async def post_init(application):
    """Registers the complete 9-command list to the Telegram Menu."""
    commands = [
        BotCommand("start", "Introduction and full manual"),
        BotCommand("myfiles", "Explore files & quick actions"),
        BotCommand("upload", "Save & push file to GitHub"),
        BotCommand("run", "Start bot from bot.json"),
        BotCommand("stop", "Kill a running process [slug]"),
        BotCommand("logs", "View blockquote logs [slug]"),
        BotCommand("deployments", "List active tasks"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Permanently delete a file")
    ]
    await application.bot.set_my_commands(commands)

# --- Logic: Dynamic Execution & Logs ---

async def get_logs_view(uid, slug):
    """Generates the blockquote log view with interactive buttons."""
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Empty log file."

    # Blockquote formatting: > at start of each line
    formatted_logs = "\n".join([f">{line}" for line in log_content.split("\n")])
    text = f"📋 *Logs for:* `{escape_md(slug)}`\n\n{escape_md(formatted_logs)}"
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

async def dynamic_run(update, context, filename):
    """Determines how to run the file based on its extension."""
    uid = update.effective_user.id
    user_path = engine.get_user_base(uid)
    ext = filename.split('.')[-1].lower()
    slug = filename.split('.')[0]
    pid = f"{uid}_{slug}"
    
    runtimes = {
        "py": f"{engine.get_venv_exe(uid)} {filename}",
        "js": f"node {filename}",
        "sh": f"bash {filename}"
    }
    
    cmd = runtimes.get(ext)
    if not cmd:
        return await update.effective_message.reply_text(f"❌ No runtime for `.{ext}` files.")

    log_p = os.path.join(user_path, f"{pid}.log")
    open(log_p, 'w').close()
    
    proc = subprocess.Popen(cmd, shell=True, cwd=user_path, stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    
    text, markup = await get_logs_view(uid, slug)
    await update.effective_message.reply_text(f"🚀 *Started `{escape_md(filename)}`*\n\n{text}", reply_markup=markup, parse_mode="MarkdownV2")

# --- Command Handlers ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v36\.0*" + "\n"
        r"Auto\-syncing to GitHub is enabled\." + "\n\n"
        r"📑 *COMPLETE COMMAND LIST*" + "\n"
        r"• `/start` — Intro \& command list\." + "\n"
        r"• `/myfiles` — Explorer with Run/Delete buttons\." + "\n"
        r"• `/upload [name]` — Reply to file to save \& push\." + "\n"
        r"• `/run` — Start via `bot.json` config\." + "\n"
        r"• `/stop [slug]` — Kill a bot process\." + "\n"
        r"• `/logs [slug]` — Blockquote logs with Refresh\." + "\n"
        r"• `/deployments` — List active bot tasks\." + "\n"
        r"• `/send [slug] [txt]` — Terminal input\." + "\n"
        r"• `/delete [name]` — Remove file locally\."
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
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
    success, err = engine.git_push_file(uid, fname)
    status = "✅ Saved & Pushed to GitHub" if success else f"⚠️ Local Only (Git Error: {escape_md(err)})"
    await update.message.reply_text(status, parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    msg = "🛰 *Active Bots:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else "📭 No active tasks."
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    
    if data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "myfiles":
        files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
        kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
        kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
        await query.edit_message_text("📂 *Your Lab Files:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    
    elif data.startswith("fopt_"):
        fname = data.replace("fopt_", ""); escaped_fname = escape_md(fname)
        row1 = [InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{fname}")]
        if fname == "requirements.txt": row1.append(InlineKeyboardButton("📦 Install Deps", callback_data="inst_deps"))
        kb = [row1, [InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}"), InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}")], [InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 *File:* `{escaped_fname}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("qrun_"): await dynamic_run(update, context, data.replace("qrun_", ""))
    elif data == "inst_deps":
        venv_pip = engine.get_venv_exe(uid).replace("python3", "pip")
        subprocess.Popen(f"{venv_pip} install -r requirements.txt", shell=True, cwd=engine.get_user_base(uid))
        await query.edit_message_text("⏳ *Installing dependencies in venv...*", parse_mode="MarkdownV2")
    elif data.startswith("logref_"):
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="MarkdownV2")
        except BadRequest: pass
    elif data.startswith("stop_"):
        slug = data.replace("stop_", ""); pid = f"{uid}_{slug}"
        if pid in running_processes:
            running_processes[pid]['proc'].terminate(); del running_processes[pid]
            await query.edit_message_text(f"🛑 Stopped `{escape_md(slug)}`", parse_mode="MarkdownV2")

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # Register all handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("myfiles", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("run", lambda u,c: dynamic_run(u,c, "bot.json")))
    app.add_handler(CommandHandler("stop", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("logs", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("deployments", deployments_cmd))
    app.add_handler(CommandHandler("send", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("delete", lambda u,c: cb_handler(u,c)))
    
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
