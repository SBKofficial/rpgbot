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

def run_git_push(uid, commit_msg):
    # Get the specific user's folder path
    user_path = get_user_base(uid)
    
    # Run Git commands ONLY inside that user's directory
    try:
        subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=user_path)
        subprocess.run('git config user.name "BotLabManager"', shell=True, cwd=user_path)
        
        # Check if it's actually a git repo before pushing
        if not os.path.exists(os.path.join(user_path, ".git")):
            return subprocess.CompletedProcess(args=[], returncode=1, stderr="Not a git repo")

        subprocess.run("git add .", shell=True, cwd=user_path)
        subprocess.run(f"git commit -m '{commit_msg}'", shell=True, cwd=user_path)
        return subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=user_path)
    except Exception as e:
        logging.error(f"Git Error for {uid}: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1)

async def upload_cmd(update, context):
    user = update.effective_user
    uid = user.id
    base = get_user_base(uid) # This ensures the folder exists
    
    if not update.message.reply_to_message or not context.args: 
        return await update.message.reply_text("❌ Reply to a code message/file with `/upload filename.py`")

    replied = update.message.reply_to_message
    filename = context.args[0]
    target = os.path.join(base, filename)

    try:
        if replied.document:
            f_obj = await replied.document.get_file()
            content = (await f_obj.download_as_bytearray()).decode('utf-8')
        elif replied.text:
            content = replied.text
        else:
            return await update.message.reply_text("❌ No text or document detected.")

        with open(target, "w") as f:
            f.write(content.strip())
        
        # Pass UID to the push function
        res = run_git_push(uid, f"Upload {filename} by {user.first_name}")
        
        if res.returncode == 0:
            await update.message.reply_text(f"✅ `{escape_md(filename)}` uploaded and synced to GitHub\.")
        else:
            await update.message.reply_text(f"⚠️ `{escape_md(filename)}` saved locally in your lab, but sync failed \(GitHub error or no repo\)\.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error during upload: `{escape_md(str(e))}`")


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

# --- 1. /start (Verified Introduction) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Welcome to Bot Lab Manager v18\.9*" + "\n"
        r"Your personal cloud environment for running and managing bots\." + "\n\n"
        r"📂 *FILE MANAGEMENT*" + "\n"
        r"• `/status` — Explorer to run files or install deps\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file/text\." + "\n"
        r"• `/delete [name]` — Remove a file permanently\." + "\n"
        r"• `/sync` — Push files to GitHub\." + "\n\n"
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — List active tasks\." + "\n"
        r"• `/logs [slug]` — View process output\." + "\n\n"
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [slug] [cmd]` — Manual start\." + "\n"
        r"• `/stop [slug]` — Force kill a task\." + "\n"
        r"• `/send [slug] [text]` — Send interactive input \(OTP\)\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status & 3. /deployments (Restored Direct Commands) ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    base = get_user_base(uid)
    files = sorted(os.listdir(base))
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in files if not f.endswith(".log") and f != ".git"]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    text = "📂 *Explorer*"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- UI Callback ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    
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
async def send_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug = update.effective_user.id, context.args[0]
    # Smart matching for slugs (finds pirate even if it's user_RUN_pirate.py)
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if target_pid:
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

async def stop_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if target_pid: running_processes[target_pid].terminate(); await update.message.reply_text("🛑 Stopped.")

async def logs_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), f"{uid}_{slug}")
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{target_pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{target_pid}")]]
    await update.message.reply_text(f"📄 *Logs:* \n{get_formatted_logs(uid, target_pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target): os.remove(target); await update.message.reply_text("🗑 Deleted.")

if __name__ == '__main__':
    app = ApplicationBuilder().token(os.getenv("BOT_TOKEN")).build()
    handlers = [("start", start_cmd), ("upload", upload_cmd), ("run", run_cmd), ("send", send_cmd), 
                ("sync", sync_cmd), ("status", status_cmd), ("deployments", deployments_cmd),
                ("stop", stop_cmd), ("logs", logs_cmd), ("delete", delete_cmd)]
    for n, f in handlers: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
