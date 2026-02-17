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
    """Strictly escapes reserved characters for Telegram MarkdownV2."""
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

# --- Command Registration ---
async def post_init(application):
    """Auto-registers all 9 commands to the Telegram Menu button on startup."""
    commands = [
        BotCommand("start", "Introduction and help guide"),
        BotCommand("myfiles", "List files and quick actions"),
        BotCommand("upload", "Upload file and push to GitHub"),
        BotCommand("run", "Run bot from bot.json"),
        BotCommand("stop", "Stop a bot [slug]"),
        BotCommand("logs", "View blockquote logs [slug]"),
        BotCommand("deployments", "List all active bots"),
        BotCommand("send", "Terminal input [slug] [txt]"),
        BotCommand("delete", "Delete a file [name]")
    ]
    await application.bot.set_my_commands(commands)

# --- UI Helpers ---
async def get_logs_view(uid, slug):
    """Generates the blockquote log view with interactive buttons."""
    path = os.path.join(engine.get_user_base(uid), f"{uid}_{slug}.log")
    log_content = "Waiting for terminal output..."
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            log_content = "".join(lines).strip() if lines else "Log file is empty."

    # YES, I REMEMBER: Blockquote formatting starts each line with '>'
    formatted_logs = "\n".join([f">{line}" for line in log_content.split("\n")])
    text = f"📋 *Logs for:* `{escape_md(slug)}`\n\n{escape_md(formatted_logs)}"
    
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logref_{slug}"),
           InlineKeyboardButton("🛑 Stop", callback_data=f"stop_{slug}")],
          [InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    return text, InlineKeyboardMarkup(kb)

# --- Command Logic ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v34\.0*" + "\n"
        r"Every `/upload` auto\-syncs to your GitHub branch\." + "\n\n"
        r"📑 *COMMAND GUIDE*" + "\n"
        r"• `/start` — Show this manual\." + "\n"
        r"• `/myfiles` — Explore files with Run/Delete buttons\." + "\n"
        r"• `/upload [name]` — Save \& push file to GitHub\." + "\n"
        r"• `/run` — Execute bot based on `bot.json`\." + "\n"
        r"• `/stop [slug]` — Stop a running process\." + "\n"
        r"• `/logs [slug]` — Blockquote logs with Refresh\." + "\n"
        r"• `/deployments` — List all active bot tasks\." + "\n"
        r"• `/send [slug] [txt]` — Send input to terminal\." + "\n"
        r"• `/delete [name]` — Remove a file permanently\."
    )
    kb = [[InlineKeyboardButton("📂 My Files", callback_data="myfiles"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    try:
        if update.callback_query:
            await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        else:
            await update.effective_message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    except BadRequest: pass

async def upload_cmd(update, context):
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Usage: Reply to a file with `/upload name.py` ")
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

async def deployments_cmd(update, context):
    uid = update.effective_user.id
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(f"{uid}_")]
    header = escape_md("🛰 Active Bots:")
    msg = f"{header}\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else escape_md("📭 No active tasks.")
    kb = [[InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    
    if update.callback_query:
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def stop_cmd(update, context):
    if not context.args: return
    slug = context.args[0]; pid = f"{update.effective_user.id}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate(); del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{escape_md(slug)}`", parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return
    fname = context.args[0]; path = os.path.join(engine.get_user_base(update.effective_user.id), fname)
    if os.path.exists(path):
        os.remove(path)
        await update.message.reply_text(f"🗑 Deleted `{escape_md(fname)}`", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to terminal.")

# --- Callback Handler ---

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
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"run_logic_{fname}"),
               InlineKeyboardButton("📋 Logs", callback_data=f"logref_{fname.split('.')[0]}")],
              [InlineKeyboardButton("🗑 Delete", callback_data=f"fdel_{fname}"),
               InlineKeyboardButton("⬅️ Back", callback_data="myfiles")]]
        await query.edit_message_text(f"📄 *File:* `{escaped_fname}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
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
    
    # 9 Command Registrations
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("myfiles", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("run", run_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_handler(CommandHandler("logs", lambda u,c: cb_handler(u,c)))
    app.add_handler(CommandHandler("deployments", deployments_cmd))
    app.add_handler(CommandHandler("send", send_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    
    app.add_handler(CallbackQueryHandler(cb_handler))
    app.run_polling()
