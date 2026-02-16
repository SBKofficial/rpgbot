import os, subprocess, logging, re, io, time, asyncio, shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

GIT_TOKEN = os.getenv("GIT_TOKEN")
BOT_TOKEN = os.getenv("BOT_TOKEN")
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
            return "\n".join([f"> {escape_md(line.strip())}" for line in lines]) if lines else r"_Log is empty\._"
    except: return r"❌ _Error reading logs\._"

# --- Workspace & Branch Isolation Logic ---
def initialize_user_workspace(uid):
    user_path = get_user_base(uid)
    branch_name = f"user_{uid}"
    if not REPO_URL: return False
    if not os.path.exists(os.path.join(user_path, ".git")):
        try:
            subprocess.run("git init", shell=True, cwd=user_path)
            subprocess.run(f"git remote add origin {REPO_URL}", shell=True, cwd=user_path)
            subprocess.run(f"git fetch origin {branch_name}", shell=True, cwd=user_path)
            subprocess.run(f"git checkout -b {branch_name}", shell=True, cwd=user_path)
            return True
        except: return False
    return False

def run_git_push(uid, commit_msg):
    user_path = get_user_base(uid)
    branch_name = f"user_{uid}"
    if not REPO_URL: return subprocess.CompletedProcess(args=[], returncode=1)
    try:
        if not os.path.exists(os.path.join(user_path, ".git")): initialize_user_workspace(uid)
        subprocess.run(f"git checkout {branch_name}", shell=True, cwd=user_path)
        subprocess.run('git config user.email "bot@lab.com"', shell=True, cwd=user_path)
        subprocess.run('git config user.name "BotLabManager"', shell=True, cwd=user_path)
        subprocess.run(f"git pull origin {branch_name}", shell=True, cwd=user_path)
        subprocess.run("git add .", shell=True, cwd=user_path)
        check = subprocess.run("git status --porcelain", shell=True, cwd=user_path, capture_output=True, text=True)
        if not check.stdout.strip(): return subprocess.CompletedProcess(args=[], returncode=0)
        subprocess.run(f"git commit -m '{commit_msg}'", shell=True, cwd=user_path)
        return subprocess.run(f"git push origin {branch_name}", shell=True, capture_output=True, cwd=user_path)
    except: return subprocess.CompletedProcess(args=[], returncode=1)

def start_unbuffered_proc(cmd, cwd, log_path):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{cwd}:{env.get('PYTHONPATH', '')}"
    return subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)

async def monitor_process(context, uid, pid, slug):
    proc = running_processes.get(pid)
    if not proc: return
    while proc.poll() is None: await asyncio.sleep(2)
    if pid in running_processes: del running_processes[pid]
    await context.bot.send_message(uid, f"🏁 *Task Finished:* `{escape_md(slug)}`", parse_mode="MarkdownV2")

# --- Commands ---
async def start_cmd(update, context):
    uid = update.effective_user.id
    initialize_user_workspace(uid)
    msg = (
        r"🤖 *Bot Lab Manager v20\.1*" + "\n"
        r"📦 *ISOLATED GIT*" + "\n"
        r"• `/clone [url]` — Clone repo to your folder\." + "\n"
        r"• `/sync` — Push to your private branch\." + "\n\n"
        r"📂 *LAB CONTROL*" + "\n"
        r"• `/status` — File explorer \(Private\)\." + "\n"
        r"• `/upload [name]` — Save code from reply\." + "\n"
        r"• `/search [query]` — Recursive file search\." + "\n"
        r"• `/delete [name]` — Remove file/folder\." + "\n\n"
        r"▶️ *EXECUTION*" + "\n"
        r"• `/run [slug] [cmd]` — Start manual process\." + "\n"
        r"• `/send [slug] [text]` — Input for OTPs\." + "\n"
        r"• `/deployments` — List active tasks\." + "\n"
        r"• `/logs [slug]` — View process output\." + "\n"
        r"• `/stop [slug]` — Kill a task\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"), InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def status_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    all_files = [os.path.relpath(os.path.join(r, f), base) for r, d, files in os.walk(base) if ".git" not in r for f in files if not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in sorted(all_files)[:15]]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    if update.callback_query: await update.callback_query.edit_message_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def search_cmd(update, context):
    if not context.args: return
    uid, query, base = update.effective_user.id, context.args[0].lower(), get_user_base(update.effective_user.id)
    results = [os.path.relpath(os.path.join(r, f), base) for r, d, files in os.walk(base) if ".git" not in r for f in files if query in f.lower()]
    if not results: return await update.message.reply_text("❌ No files found.")
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in results[:10]]
    await update.message.reply_text(f"🔍 *Results:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()
    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await query.edit_message_text("🏠 *Home Menu*", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"), InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]), parse_mode="MarkdownV2")
    elif data == "view_deploys":
        prefix = f"{uid}_"
        procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
        msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]), parse_mode="MarkdownV2")
    elif data.startswith("manage_"):
        f_path = data.replace("manage_", ""); f_name = os.path.basename(f_path); pid = f"{uid}_{f_name}"
        kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{f_path}"), InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}")]]
        kb.append([InlineKeyboardButton("🗑 Delete", callback_data=f"uidelete_{f_path}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")])
        await query.edit_message_text(f"📄 *File:* `{escape_md(f_path)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    elif data.startswith("qrun_"):
        f_path = data.replace("qrun_", ""); base = get_user_base(uid); full_p = os.path.join(base, f_path)
        f_name = os.path.basename(f_path); pid = f"{uid}_{f_name}"; log_p = os.path.join(base, f"{pid}.log")
        cmd = f"python3 -u {f_name}" if f_name.endswith(".py") else f"node {f_name}"
        running_processes[pid] = start_unbuffered_proc(cmd, os.path.dirname(full_p), log_p)
        asyncio.create_task(monitor_process(context, uid, pid, f_name))
        await query.edit_message_text(f"🚀 *Started:* `{f_name}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📄 Logs", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")]]), parse_mode="MarkdownV2")
    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", ""); await query.edit_message_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")
    elif data.startswith("kill_"):
        pid = data.replace("kill_", ""); (running_processes[pid].terminate(), running_processes.pop(pid)) if pid in running_processes else None
        await query.edit_message_text("🛑 Terminated\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")
    elif data.startswith("uidelete_"):
        f = data.replace("uidelete_", ""); target = os.path.join(get_user_base(uid), f)
        (shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)) if os.path.exists(target) else None
        await query.edit_message_text(f"🗑 Deleted `{f}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return await update.message.reply_text("❌ Reply with `/upload filename.py`")
    f_name = context.args[0]; target = os.path.join(base, f_name); replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()).decode('utf-8') if replied.document else replied.text
    with open(target, "w") as f: f.write(content.strip())
    res = run_git_push(uid, f"Upload {f_name}")
    await update.message.reply_text(f"✅ Saved & Synced to branch `user_{uid}`\." if res.returncode == 0 else "⚠️ Saved locally, sync failed\.", parse_mode="MarkdownV2")

async def sync_cmd(update, context):
    uid = update.effective_user.id; res = run_git_push(uid, "Manual Sync")
    await update.message.reply_text(f"✅ Sync Complete \(Branch: `user_{uid}`\)" if res.returncode == 0 else "❌ Sync Failed", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug, text = update.effective_user.id, context.args[0], " ".join(context.args[1:])
    pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if pid: running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush(); await update.message.reply_text("⌨️ Sent to stdin\.")

async def stop_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), None)
    if pid: running_processes[pid].terminate(); running_processes.pop(pid); await update.message.reply_text(f"🛑 Stopped `{slug}`\.")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    await update.message.reply_text("🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\.", parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    pid = next((p for p in running_processes if p.startswith(str(uid)) and slug in p), f"{uid}_{slug}")
    await update.message.reply_text(f"📄 *Logs for {slug}:*\n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target): (shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)); await update.message.reply_text("🗑 Deleted\.")

async def clone_cmd(update, context):
    if not context.args: return
    uid, url = update.effective_user.id, context.args[0]
    folder = url.split("/")[-1].replace(".git", ""); target = os.path.join(get_user_base(uid), folder)
    res = subprocess.run(f"git clone {url} {target}", shell=True, capture_output=True)
    await update.message.reply_text("✅ Cloned\!" if res.returncode == 0 else "❌ Failed\.")

async def run_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug, cmd = update.effective_user.id, context.args[0], " ".join(context.args[1:])
    pid = f"{uid}_{slug}"; log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = start_unbuffered_proc(cmd, get_user_base(uid), log_p)
    asyncio.create_task(monitor_process(context, uid, pid, slug))
    await update.message.reply_text(f"🚀 Started `{slug}`")

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    cmds = [("start", start_cmd), ("status", status_cmd), ("search", search_cmd), ("clone", clone_cmd), ("run", run_cmd), ("send", send_cmd), ("upload", upload_cmd), ("sync", sync_cmd), ("stop", stop_cmd), ("deployments", deployments_cmd), ("logs", logs_cmd), ("delete", delete_cmd)]
    for n, f in cmds: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
