import os, subprocess, logging, re, io, time
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

GIT_TOKEN = os.getenv("GIT_TOKEN")
REPO_URL = f"https://{GIT_TOKEN}@github.com/SBKofficial/rpgbot.git" if GIT_TOKEN else None

running_processes = {} 
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- Utilities ---
def escape_md(text): 
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_user_base(uid): 
    path = os.path.join(LAB_DIR, str(uid))
    os.makedirs(path, exist_ok=True)
    return path

def get_formatted_logs(uid, pid):
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ _No logs available yet\._"
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            if not lines: return r"_Log is currently empty\._"
            return "\n".join([f"> {escape_md(line.strip())}" for line in lines])
    except: return r"❌ _Error reading logs\._"

def run_git_push(commit_msg):
    subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=ROOT_DIR)
    subprocess.run('git config user.name "BotLabManager"', shell=True, cwd=ROOT_DIR)
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run(f"git commit -m '{commit_msg}'", shell=True, cwd=ROOT_DIR)
    return subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)

# --- 1. /start ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v17\.7*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"📂 *FILE COMMANDS*" + "\n"
        r"• `/status` — Explorer to run/delete/install deps\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file or text\." + "\n"
        r"• `/sync` — Push all files to GitHub repository\." + "\n\n"
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — View active processes\." + "\n"
        r"• `/logs [name]` — View output in blockquote format\." + "\n\n"
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [slug] [cmd]` — Start a process manually\." + "\n"
        r"• `/stop [slug]` — Kill a process\." + "\n"
        r"• `/send [slug] [text]` — Send text to a bot's stdin\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    
    if update.callback_query:
        try: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status & 3. /deployments ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    base = get_user_base(uid)
    files = sorted(os.listdir(base))
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in files if not f.endswith(".log") and f != ".git"]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    text = f"📂 *Explorer:* `{escape_md(os.path.basename(base))}`"
    if update.callback_query:
        try: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 4. /upload (Fixed AttributeError and UnboundLocalError) ---
async def upload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    base = get_user_base(uid)
    
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ Reply to a file or code block with: `/upload name.py`")
    
    replied = update.message.reply_to_message
    filename = context.args[0]
    target = os.path.join(base, filename)
    
    # FIXED: Check for document first, then text to avoid NoneType.strip() crash
    content = ""
    if replied.document:
        doc_file = await replied.document.get_file()
        byte_array = await doc_file.download_as_bytearray()
        content = byte_array.decode('utf-8')
    elif replied.text:
        content = replied.text
    else:
        return await update.message.reply_text("❌ Replied message has no readable content.")

    with open(target, "w") as f: 
        f.write(content.strip())
    
    m = await update.message.reply_text("💾 Saved. Syncing...")
    res = run_git_push(f"User {uid} uploaded {filename}")
    if res.returncode == 0: await m.edit_text(f"✅ `{escape_md(filename)}` pushed to GitHub!")
    else: await m.edit_text(f"⚠️ Saved locally, Push failed: `{escape_md(res.stderr[:40])}`")

# --- 5. /run, 6. /stop, 7. /logs, 8. /send, 9. /sync ---
async def run_cmd(update, context):
    if len(context.args) < 2: return
    uid = update.effective_user.id
    pid = f"{uid}_{context.args[0]}"
    cmd = " ".join(context.args[1:])
    log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
    await update.message.reply_text(f"🚀 Started `{escape_md(context.args[0])}`")

async def stop_cmd(update, context):
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid].terminate(); del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{escape_md(context.args[0])}`")

async def logs_cmd(update, context):
    if not context.args: return
    uid = update.effective_user.id
    pid = f"{uid}_{context.args[0]}"
    await update.message.reply_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    text = " ".join(context.args[1:])
    if pid in running_processes:
        running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush()
        await update.message.reply_text(f"⌨️ Sent to `{escape_md(context.args[0])}`")

async def sync_cmd(update, context):
    m = await update.message.reply_text("🔄 Syncing Lab...")
    res = run_git_push("Manual Lab Sync")
    if res.returncode == 0: await m.edit_text("✅ GitHub Sync OK")
    else: await m.edit_text(f"❌ Failed: {escape_md(res.stderr[:50])}")

# --- UI Callback Logic ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    
    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Quick Run", callback_data=f"qrun_{f}"), InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}")]]
        if "requirements" in f.lower(): kb.append([InlineKeyboardButton("📦 Install Deps", callback_data=f"pipinst_{f}")])
        kb.append([InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")])
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("pipinst_"):
        f = data.replace("pipinst_", ""); pid = f"{uid}_pip_install"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        subprocess.Popen(f"pip install -r {f}", shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📦 *Installing\.\.\.*\n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{f}"
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"🚀 *Running:* `{escape_md(f)}` \n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        try: await query.edit_message_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 Terminated.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    for name, func in [("start", start_cmd), ("upload", upload_cmd), ("status", status_cmd), 
                        ("deployments", deployments_cmd), ("run", run_cmd), ("stop", stop_cmd), 
                        ("logs", logs_cmd), ("send", send_cmd), ("sync", sync_cmd)]:
        app.add_handler(CommandHandler(name, func))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
