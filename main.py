import os, subprocess, logging, re, asyncio, json, shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import BadRequest
from engine import LabEngine

# --- Initialization ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, slug} }

logging.basicConfig(level=logging.INFO)

def escape_md(text):
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# --- Command Registration ---
async def post_init(application):
    """Registers the 9 commands to the Telegram Menu button."""
    commands = [
        BotCommand("start", "Intro & manual"),
        BotCommand("upload", "Save & push [name]"),
        BotCommand("myfiles", "File explorer & Run"),
        BotCommand("run", "Start from bot.json"),
        BotCommand("stop", "Stop bot [slug]"),
        BotCommand("logs", "View logs [slug]"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("deployments", "List active bots"),
        BotCommand("delete", "Delete file [name]")
    ]
    await application.bot.set_my_commands(commands)

# --- UI Logic ---
async def get_logs_view(uid, slug):
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for logs..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Empty log file."

    formatted_logs = "\n".join([f">{line}" for line in log_content.split("\n")])
    text = f"📋 *Logs for:* `{escape_md(slug)}`\n\n{escape_md(formatted_logs)}"
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")]]
    return text, InlineKeyboardMarkup(kb)

# --- Handlers ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v31\.0*" + "\n"
        r"Every upload is synced to your GitHub branch\." + "\n\n"
        r"📑 *COMPLETE COMMAND LIST*" + "\n"
        r"• `/upload [name]` — Save \& auto\-push file\." + "\n"
        r"• `/myfiles` — Explore files with Run/Delete buttons\." + "\n"
        r"• `/run` — Execute bot based on `bot.json`\." + "\n"
        r"• `/logs [slug]` — Blockquote log view with Refresh\." + "\n"
        r"• `/stop [slug]` — Kill a running process\." + "\n"
        r"• `/send [slug] [txt]` — Send terminal input\." + "\n"
        r"• `/deployments` — List all active tasks\." + "\n"
        r"• `/delete [name]` — Remove a file permanently\."
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file with `/upload name.py` ")
    uid, fname = update.effective_user.id, context.args[0]
    path = os.path.join(engine.get_user_base(uid), fname)
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()) if replied.document else replied.text.encode()
    with open(path, "wb") as f: f.write(content)
    success, err = engine.git_push_file(uid, fname)
    status = "✅ Saved & Pushed to GitHub" if success else f"⚠️ Saved Locally (Git Error: {escape_md(err)})"
    await update.message.reply_text(status, parse_mode="MarkdownV2")

async def run_cmd(update, context):
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config: return await update.message.reply_text("❌ `bot.json` missing.")
    slug = config['name']; pid = f"{uid}_{slug}"
    log_p = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    if os.path.exists(log_p): open(log_p, 'w').close()
    
    proc = subprocess.Popen(config['start_cmd'].replace("python3", engine.get_venv_exe(uid)), 
                            shell=True, cwd=engine.get_user_base(uid), stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    running_processes[pid] = {"proc": proc, "slug": slug}
    text, markup = await get_logs_view(uid, slug)
    await update.effective_message.reply_text(f"🚀 *Started\!*\n\n{text}", reply_markup=markup, parse_mode="MarkdownV2")

async def stop_cmd(update, context):
    if not context.args: return
    slug = context.args[0]; pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate(); del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{slug}`")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Input sent.")

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id; await query.answer()
    
    if data == "nav_home": await start_cmd(update, context)
    elif data == "myfiles":
        files = [f for f in os.listdir(engine.get_user_base(uid)) if f not in ["venv", ".git"] and not f.endswith(".log")]
        kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"fopt_{f}")] for f in sorted(files)]
        kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
        await query.edit_message_text("📂 *Your Lab Files:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    elif data.startswith("fopt_"):
        fname = data.replace("fopt_", "")
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"run_logic_{fname}"),
               InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}")],
              [InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}"),
               InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(fname)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    elif data.startswith("logref_"):
        text, markup = await get_logs_view(uid, data.replace("logref_", ""))
        try: await query.edit_message_text(text, reply_markup=markup, parse_mode="MarkdownV2")
        except BadRequest: pass
    elif data == "view_deploys":
        active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
        kb = [[InlineKeyboardButton(f"✅ {p}", callback_data=f"logref_{p}")] for p in active]
        kb.append([InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
        await query.edit_message_text("🛰 *Active Tasks:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()
    
    # Registering all Command Handlers
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("myfiles", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("logs", lambda u,c: run_cmd(u,c))) # Uses run logic to show logs
    app.add_handler(CommandHandler("send", send_cmd))
    app.add_handler(CommandHandler("deployments", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("delete", lambda u,c: cb_handler(u,c)))
    
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
