import os, subprocess, logging, re, io, time, asyncio
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

async def monitor_process(context, uid, pid, slug):
    """Wait for process to finish and then push final logs to user"""
    proc = running_processes.get(pid)
    if not proc: return
    while proc.poll() is None: await asyncio.sleep(2)
    if pid in running_processes: del running_processes[pid]
    logs = get_formatted_logs(uid, pid)
    await context.bot.send_message(uid, f"🏁 *Process Finished:* `{escape_md(slug)}`\n\n*Final Logs:*\n{logs}", parse_mode="MarkdownV2")

# --- 1. /start ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v18\.0*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"📂 *FILE COMMANDS*" + "\n"
        r"• `/status` — Explorer to run/delete/install deps\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file\." + "\n"
        r"• `/delete [name]` — Remove a file permanently\." + "\n"
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
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status & 3. /deployments ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    base = get_user_base(uid)
    files = sorted(os.listdir(base))
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in files if not f.endswith(".log") and f != ".git"]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    text = f"📂 *Explorer:* `{escape_md(os.path.basename(base))}`"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 4. /run (Directly shows logs) ---
async def run_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug = update.effective_user.id, context.args[0]
    pid = f"{uid}_{slug}"
    cmd = " ".join(context.args[1:])
    log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
    asyncio.create_task(monitor_process(context, uid, pid, slug))
    await update.message.reply_text(f"🚀 *Started:* `{escape_md(slug)}` \n\n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

# --- UI Callback (Restored Install Deps & Run Logs) ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{f}"), InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}")]]
        if "requirements" in f.lower(): kb.append([InlineKeyboardButton("📦 Install Deps", callback_data=f"pipinst_{f}")])
        kb.append([InlineKeyboardButton("🗑 Delete", callback_data=f"uidelete_{f}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")])
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("pipinst_"):
        f = data.replace("pipinst_", ""); pid = f"{uid}_pip_install"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = subprocess.Popen(f"pip install -r {f}", shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
        asyncio.create_task(monitor_process(context, uid, pid, "PIP Install"))
        await query.edit_message_text(f"📦 *Installing\.\.\.*\n\n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{f}"
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
        asyncio.create_task(monitor_process(context, uid, pid, f))
        await query.edit_message_text(f"🚀 *Running:* `{escape_md(f)}` \n\n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", "")
        await query.edit_message_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")

    elif data.startswith("uidelete_"):
        f = data.replace("uidelete_", ""); target = os.path.join(get_user_base(uid), f)
        if os.path.exists(target): os.remove(target)
        await query.edit_message_text(f"🗑 Deleted `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

# --- Remaining Commands ---
async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return await update.message.reply_text("❌ Reply to a file with: `/upload name.py`")
    replied = update.message.reply_to_message; filename = context.args[0]; target = os.path.join(base, filename)
    if replied.document:
        f_obj = await replied.document.get_file(); b_arr = await f_obj.download_as_bytearray(); content = b_arr.decode('utf-8')
    elif replied.text: content = replied.text
    else: return await update.message.reply_text("❌ No content.")
    with open(target, "w") as f: f.write(content.strip())
    m = await update.message.reply_text("💾 Saved. Syncing...")
    res = run_git_push(f"User {uid} uploaded {filename}")
    if res.returncode == 0: await m.edit_text(f"✅ `{escape_md(filename)}` pushed!")
    else: await m.edit_text(f"⚠️ Saved locally, Push failed.")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target): os.remove(target); await update.message.reply_text(f"🗑 Deleted `{escape_md(context.args[0])}`")

async def stop_cmd(update, context):
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes: running_processes[pid].terminate(); await update.message.reply_text(f"🛑 Stopped `{escape_md(context.args[0])}`")

async def logs_cmd(update, context):
    if not context.args: return
    await update.message.reply_text(f"📄 *Logs:* \n{get_formatted_logs(update.effective_user.id, f'{update.effective_user.id}_{context.args[0]}')}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes: running_processes[pid].stdin.write(context.args[1] + "\n"); running_processes[pid].stdin.flush(); await update.message.reply_text("⌨️ Sent.")

async def sync_cmd(update, context):
    m = await update.message.reply_text("🔄 Syncing...")
    res = run_git_push("Manual Sync")
    await m.edit_text("✅ Sync OK" if res.returncode == 0 else "❌ Sync Failed")

if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv("BOT_TOKEN")).build()
    handlers = [("start", start_cmd), ("upload", upload_cmd), ("status", status_cmd), ("deployments", deployments_cmd), 
                ("delete", delete_cmd), ("run", run_cmd), ("stop", stop_cmd), ("logs", logs_cmd), ("send", send_cmd), ("sync", sync_cmd)]
    for n, f in handlers: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
