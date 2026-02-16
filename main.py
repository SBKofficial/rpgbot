import os, subprocess, logging, re, io, time, asyncio, shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

# Ensure your GIT_TOKEN is set in your environment variables
GIT_TOKEN = os.getenv("GIT_TOKEN")
# This is your master repo where everyone's files are backed up
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

# --- FIXED: Auto-Initializing Git Push ---
def run_git_push(uid, commit_msg):
    user_path = get_user_base(uid)
    if not REPO_URL:
        return subprocess.CompletedProcess(args=[], returncode=1, stderr=b"GIT_TOKEN not set")

    try:
        # Check if .git exists; if not, initialize it
        if not os.path.exists(os.path.join(user_path, ".git")):
            subprocess.run("git init", shell=True, cwd=user_path)
            subprocess.run(f"git remote add origin {REPO_URL}", shell=True, cwd=user_path)
            # Pull first to ensure we don't have conflicts with the main repo
            subprocess.run("git pull origin main", shell=True, cwd=user_path)

        subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=user_path)
        subprocess.run('git config user.name "BotLabManager"', shell=True, cwd=user_path)
        subprocess.run("git add .", shell=True, cwd=user_path)
        
        # Check if there are actual changes to commit to avoid "nothing to commit" error
        check_status = subprocess.run("git status --porcelain", shell=True, cwd=user_path, capture_output=True, text=True)
        if not check_status.stdout.strip():
            return subprocess.CompletedProcess(args=[], returncode=0) # Nothing to push, but not an error

        subprocess.run(f"git commit -m '{commit_msg}'", shell=True, cwd=user_path)
        return subprocess.run(f"git push origin main", shell=True, capture_output=True, cwd=user_path)
    except Exception as e:
        logging.error(f"Git Error for {uid}: {e}")
        return subprocess.CompletedProcess(args=[], returncode=1, stderr=str(e).encode())

def start_unbuffered_proc(cmd, cwd, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NODE_OPTIONS"] = "--unbuffered"
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    return subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, 
        stdin=subprocess.PIPE, text=True, bufsize=0
    )

async def monitor_process(context, uid, pid, slug):
    proc = running_processes.get(pid)
    if not proc: return
    while proc.poll() is None: await asyncio.sleep(2)
    if pid in running_processes: del running_processes[pid]
    logs = get_formatted_logs(uid, pid)
    await context.bot.send_message(uid, f"🏁 *Process Finished:* `{escape_md(slug)}`\n\n*Final Logs:*\n{logs}", parse_mode="MarkdownV2")

# --- Commands ---
async def start_cmd(update, context):
    msg = (
        r"🤖 *Bot Lab Manager v19\.6*" + "\n"
        r"Your introduction to the bot's features:" + "\n\n"
        r"📦 *GIT & REPOS*" + "\n"
        r"• `/clone [url]` — Clone any repo into your space\." + "\n"
        r"• `/sync` — Push your current lab to GitHub\." + "\n\n"
        r"📂 *LAB CONTROL*" + "\n"
        r"• `/status` — File explorer with module support\." + "\n"
        r"• `/search [query]` — Find files in sub\-folders\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file\." + "\n"
        r"• `/delete [name]` — Remove a file or folder\." + "\n\n"
        r"▶️ *EXECUTION*" + "\n"
        r"• `/run [slug] [cmd]` — Start a custom command\." + "\n"
        r"• `/send [slug] [text]` — Input for OTPs/prompts\." + "\n"
        r"• `/deployments` — List your active tasks\." + "\n"
        r"• `/stop [slug]` — Kill a running task\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def status_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    all_files = []
    for root, dirs, files in os.walk(base):
        if ".git" in root or "node_modules" in root: continue
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), base)
            if not f.endswith(".log"): all_files.append(rel)
    
    all_files = sorted(all_files)[:15]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in all_files]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    text = "📂 *Explorer*"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def search_cmd(update, context):
    if not context.args: return
    uid, query, base = update.effective_user.id, context.args[0].lower(), get_user_base(update.effective_user.id)
    results = []
    for root, dirs, files in os.walk(base):
        if ".git" in root: continue
        for f in files:
            if query in f.lower(): results.append(os.path.relpath(os.path.join(root, f), base))
    if not results: return await update.message.reply_text("❌ No files found.")
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in results[:10]]
    await update.message.reply_text(f"🔍 *Results for:* `{query}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys":
        prefix = f"{uid}_"
        procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
        msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="MarkdownV2")
    
    elif data.startswith("manage_"):
        f_path = data.replace("manage_", ""); f_name = os.path.basename(f_path); pid = f"{uid}_{f_name}"
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{f_path}"), InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}")]]
        if "requirements.txt" in f_path: kb.append([InlineKeyboardButton("📦 Install Py Deps", callback_data=f"pipinst_{f_path}")])
        if "package.json" in f_path: kb.append([InlineKeyboardButton("📦 Install Node Deps", callback_data=f"npminst_{f_path}")])
        kb.append([InlineKeyboardButton("🗑 Delete", callback_data=f"uidelete_{f_path}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")])
        await query.edit_message_text(f"📄 *File:* `{escape_md(f_path)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif any(data.startswith(x) for x in ["qrun_", "pipinst_", "npminst_"]):
        mode = "RUN"
        if "pipinst_" in data: mode, f_path = "PIP", data.replace("pipinst_", "")
        elif "npminst_" in data: mode, f_path = "NPM", data.replace("npminst_", "")
        else: f_path = data.replace("qrun_", "")
        
        full_p = os.path.join(get_user_base(uid), f_path)
        module_root = os.path.dirname(full_p)
        f_name = os.path.basename(f_path)
        pid = f"{uid}_{mode}_{f_name}"
        
        if mode == "PIP": cmd = f"pip install -r {f_name}"
        elif mode == "NPM": cmd = "npm install"
        else: cmd = f"node {f_name}" if f_name.endswith(".js") else f"python3 -u {f_name}"
        
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = start_unbuffered_proc(cmd, module_root, log_p)
        asyncio.create_task(monitor_process(context, uid, pid, f_name))
        await asyncio.sleep(1)
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"🚀 *Started {mode}:* `{escape_md(f_name)}` \n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

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
        if os.path.exists(target): 
            if os.path.isdir(target): shutil.rmtree(target)
            else: os.remove(target)
        await query.edit_message_text(f"🗑 Deleted `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

async def clone_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/clone [url]`")
    uid, url = update.effective_user.id, context.args[0]
    folder = url.split("/")[-1].replace(".git", "")
    target = os.path.join(get_user_base(uid), folder)
    if os.path.exists(target): return await update.message.reply_text("⚠️ Folder already exists.")
    msg = await update.message.reply_text("🛰 *Cloning repository\.\.\.*", parse_mode="MarkdownV2")
    res = subprocess.run(f"git clone {url} {target}", shell=True, capture_output=True)
    if res.returncode == 0: await msg.edit_text(f"✅ Successfully cloned `{folder}`")
    else: await msg.edit_text(f"❌ Clone failed: {res.stderr.decode()[:100]}")

async def send_cmd(update, context):
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: `/send [slug] [text]`")
    uid, slug, text = update.effective_user.id, context.args[0], " ".join(context.args[1:])
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if target_pid:
        running_processes[target_pid].stdin.write(text + "\n")
        running_processes[target_pid].stdin.flush()
        await update.message.reply_text(f"⌨️ Sent to `{slug}` stdin\.")
    else: await update.message.reply_text("❌ Process not found.")

async def run_cmd(update, context):
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: `/run [slug] [cmd]`")
    uid, slug, cmd = update.effective_user.id, context.args[0], " ".join(context.args[1:])
    pid = f"{uid}_{slug}"; log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = start_unbuffered_proc(cmd, get_user_base(uid), log_p)
    asyncio.create_task(monitor_process(context, uid, pid, slug))
    await update.message.reply_text(f"🚀 Started manual task: `{slug}`\.", parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return await update.message.reply_text("❌ Reply with `/upload filename.py`")
    replied = update.message.reply_to_message; filename = context.args[0]; target = os.path.join(base, filename)
    if replied.document:
        f_obj = await replied.document.get_file()
        content = (await f_obj.download_as_bytearray()).decode('utf-8')
    elif replied.text: content = replied.text
    else: return await update.message.reply_text("❌ No content found.")
    with open(target, "w") as f: f.write(content.strip())
    res = run_git_push(uid, f"Upload {filename}")
    await update.message.reply_text(f"✅ `{escape_md(filename)}` synced\." if res.returncode == 0 else "⚠️ Saved locally, sync skipped.")

async def sync_cmd(update, context):
    uid = update.effective_user.id
    res = run_git_push(uid, "Manual Sync")
    await update.message.reply_text("✅ Sync OK" if res.returncode == 0 else "❌ Sync Failed")

async def stop_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if target_pid: running_processes[target_pid].terminate(); await update.message.reply_text(f"🛑 Stopped `{slug}`\.")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    await update.message.reply_text(msg, parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    target_pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), f"{uid}_{slug}")
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{target_pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{target_pid}")]]
    await update.message.reply_text(f"📄 *Logs for {slug}:*\n{get_formatted_logs(uid, target_pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target):
        if os.path.isdir(target): shutil.rmtree(target)
        else: os.remove(target)
        await update.message.reply_text("🗑 Deleted\.")

if __name__ == '__main__':
    token = os.getenv("BOT_TOKEN")
    if not token: print("BOT_TOKEN not found!"); exit()
    app = ApplicationBuilder().token(token).build()
    handlers = [
        ("start", start_cmd), ("status", status_cmd), ("search", search_cmd), ("clone", clone_cmd),
        ("run", run_cmd), ("send", send_cmd), ("upload", upload_cmd), ("sync", sync_cmd),
        ("stop", stop_cmd), ("deployments", deployments_cmd), ("logs", logs_cmd), ("delete", delete_cmd)
    ]
    for n, f in handlers: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
