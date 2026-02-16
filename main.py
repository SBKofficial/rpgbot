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
    proc = running_processes.get(pid)
    if not proc: return
    while proc.poll() is None: await asyncio.sleep(2)
    if pid in running_processes: del running_processes[pid]
    logs = get_formatted_logs(uid, pid)
    await context.bot.send_message(uid, f"🏁 *Process Finished:* `{escape_md(slug)}`\n\n*Final Logs:*\n{logs}", parse_mode="MarkdownV2")

def start_unbuffered_proc(cmd, cwd, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NODE_OPTIONS"] = "--unbuffered"
    return subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, 
        stdin=subprocess.PIPE, text=True, bufsize=0
    )

# --- 1. /start (Full Guide) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Welcome to Bot Lab Manager v18\.8*" + "\n"
        r"Your personal cloud environment for running and managing bots\." + "\n\n"
        r"📂 *FILE MANAGEMENT*" + "\n"
        r"• `/status` — Open Explorer to run, delete, or install deps\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file/text\." + "\n"
        r"• `/delete [name]` — Remove a file permanently\." + "\n"
        r"• `/sync` — Push lab files to GitHub\." + "\n\n"
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — List active background tasks\." + "\n"
        r"• `/logs [slug]` — View output of a running process\." + "\n\n"
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [slug] [cmd]` — Start a process manually\." + "\n"
        r"• `/stop [slug]` — Kill a running process\." + "\n"
        r"• `/send [slug] [text]` — Send interactive input to stdin\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- UI Callback Logic ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": 
        base = get_user_base(uid)
        files = sorted(os.listdir(base))
        kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in files if not f.endswith(".log") and f != ".git"]
        kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
        await query.edit_message_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys":
        prefix = f"{uid}_"
        procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
        msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="MarkdownV2")

    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{f}"), InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}")]]
        if "requirements.txt" in f: kb.append([InlineKeyboardButton("📦 Install Py Deps", callback_data=f"pipinst_{f}")])
        if "package.json" in f: kb.append([InlineKeyboardButton("📦 Install Node Deps", callback_data=f"npminst_{f}")])
        kb.append([InlineKeyboardButton("🗑 Delete", callback_data=f"uidelete_{f}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")])
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif any(data.startswith(x) for x in ["qrun_", "pipinst_", "npminst_"]):
        mode = "RUN"
        if "pipinst_" in data: mode, f = "PIP", data.replace("pipinst_", "")
        elif "npminst_" in data: mode, f = "NPM", data.replace("npminst_", "")
        else: f = data.replace("qrun_", "")
        
        pid = f"{uid}_{mode}_{f}"
        cmd = f"pip install -r {f}" if mode == "PIP" else ("npm install" if mode == "NPM" else (f"node {f}" if f.endswith(".js") else f"python3 -u {f}"))
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = start_unbuffered_proc(cmd, get_user_base(uid), log_p)
        asyncio.create_task(monitor_process(context, uid, pid, f))
        await asyncio.sleep(1)
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"🚀 *Started {mode}:* `{escape_md(f)}` \n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        try: await query.edit_message_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 Terminated.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

    elif data.startswith("uidelete_"):
        f = data.replace("uidelete_", ""); target = os.path.join(get_user_base(uid), f)
        if os.path.exists(target): os.remove(target)
        await query.edit_message_text(f"🗑 Deleted `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

# --- Core Commands ---
async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return
    replied = update.message.reply_to_message; filename = context.args[0]; target = os.path.join(base, filename)
    if replied.document:
        f_obj = await replied.document.get_file()
        content = (await f_obj.download_as_bytearray()).decode('utf-8')
    elif replied.text: content = replied.text
    else: return
    with open(target, "w") as f: f.write(content.strip())
    res = run_git_push(f"Upload {filename}")
    await update.message.reply_text(f"✅ `{escape_md(filename)}` pushed!" if res.returncode == 0 else "⚠️ Saved locally.")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    # Search for pid matches if mode prefix exists
    target_pid = next((p for p in running_processes if p.endswith(f"_{context.args[0]}")), pid)
    if target_pid in running_processes:
        running_processes[target_pid].stdin.write(" ".join(context.args[1:]) + "\n")
        running_processes[target_pid].stdin.flush()
        await update.message.reply_text("⌨️ Sent to stdin.")

async def sync_cmd(update, context):
    res = run_git_push("Manual Sync"); await update.message.reply_text("✅ Sync OK" if res.returncode == 0 else "❌ Failed")

async def run_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug = update.effective_user.id, context.args[0]
    pid, cmd = f"{uid}_{slug}", " ".join(context.args[1:])
    log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = start_unbuffered_proc(cmd, get_user_base(uid), log_p)
    asyncio.create_task(monitor_process(context, uid, pid, slug))
    await asyncio.sleep(1)
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")]]
    await update.message.reply_text(f"🚀 *Started:* `{escape_md(slug)}` \n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv("BOT_TOKEN")).build()
    cmds = [("start", start_cmd), ("upload", upload_cmd), ("run", run_cmd), ("send", send_cmd), ("sync", sync_cmd), ("status", start_cmd), ("deployments", start_cmd)]
    for n, f in cmds: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
