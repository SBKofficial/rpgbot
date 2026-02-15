import os, subprocess, psutil, time, pathlib, logging, shutil, re, io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import telegram.error

# --- Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

GIT_TOKEN = "YOUR_GITHUB_TOKEN_HERE" 
REPO_URL = f"https://{GIT_TOKEN}@github.com/SBKofficial/rpgbot.git"

running_processes = {} 
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- Utilities ---
def escape_md(text): 
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_user_base(uid): 
    path = os.path.join(LAB_DIR, str(uid))
    os.makedirs(path, exist_ok=True)
    return path

def get_logs(uid, pid):
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ No logs found\."
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            return "\n".join([f"`{escape_md(line.strip())}`" for line in lines]) if lines else r"_Log is empty\._"
    except: return r"❌ Error reading logs\."

# --- 1. /start (The Full Introduction & Command List) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v16\.6 — Full Manual*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"👋 *Welcome\!* I am your remote bot management system\. Here is a guide to every command available to you\:" + "\n\n"
        
        r"📂 *FILE MANAGEMENT*" + "\n"
        r"• `/status` — Opens the *Explorer*\. Browse files, run them, or delete them via buttons\." + "\n"
        r"• `/upload [name]` — Save a file\. *Usage:* Reply to a code block or file with `/upload bot\.py`\." + "\n"
        r"• `/sync` — Manually pushes all local files to your GitHub repository\." + "\n\n"
        
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — Lists all bots currently running on this server\." + "\n"
        r"• `/logs [name]` — Shows a snapshot of the last 15 lines of output for a bot\." + "\n\n"
        
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [name] [cmd]` — Starts a process\. *Example:* `/run bot1 python3 bot\.py`\." + "\n"
        r"• `/stop [name]` — Instantly kills a running process\." + "\n"
        r"• `/send [name] [text]` — Sends text to a running bot \(useful for OTPs or input\)\." + "\n\n"
        
        r"🔧 *SYSTEM*" + "\n"
        r"If you cannot see files in GitHub \(403 Error\), use the *Repair* button below to refresh your token connection\."
    )
    
    kb = [
        [InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
         InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")],
        [InlineKeyboardButton("🔧 Repair GitHub Connection", callback_data="fix_git")]
    ]
    
    if update.callback_query:
        try: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except telegram.error.BadRequest: pass
    else:
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status (Explorer) ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    base = get_user_base(uid)
    kb = []
    for f in sorted(os.listdir(base)):
        if f.endswith(".log") or f == ".git": continue
        kb.append([InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")])
    
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    text = f"📂 *Explorer:* `{escape_md(os.path.basename(base))}`"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 3. /deployments ---
async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 4. /upload ---
async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args:
        return await update.message.reply_text("❌ *Error:* Reply to a file with `/upload name\.py`", parse_mode="MarkdownV2")
    
    filename = context.args[0]
    target = os.path.join(base, filename)
    replied = update.message.reply_to_message

    if replied.document:
        f_obj = await replied.document.get_file()
        out = io.BytesIO()
        await f_obj.download_to_memory(out=out)
        raw = out.getvalue().decode('utf-8')
    else: raw = replied.text

    with open(target, "w") as f: f.write(raw.strip())
    m = await update.message.reply_text("💾 Saved locally\. Syncing to GitHub\.\.\.")
    
    # Push Logic
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run(f"git commit -m 'Upload {filename}'", shell=True, cwd=ROOT_DIR)
    res = subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    
    if res.returncode == 0: await m.edit_text(f"✅ `{escape_md(filename)}` pushed to GitHub\!")
    else: await m.edit_text(f"⚠️ Saved locally, but GitHub rejected the push \(403\)\. Use the Repair button in `/start`\.")

# --- 5-9. Command Logic Group ---
async def run_cmd(update, context):
    if len(context.args) < 2: return await update.message.reply_text("❌ `/run [slug] [command]`")
    uid, pid, cmd = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
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
    uid, pid = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}"
    await update.message.reply_text(f"📄 *Logs:* {get_logs(uid, pid)}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid, text = f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
    if pid in running_processes:
        running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush()
        await update.message.reply_text(f"⌨️ Sent input to `{escape_md(context.args[0])}`")

async def sync_cmd(update, context):
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run("git commit -m 'Manual Sync'", shell=True, cwd=ROOT_DIR)
    res = subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    if res.returncode == 0: await update.message.reply_text("✅ GitHub Sync OK")
    else: await update.message.reply_text(f"❌ Sync Failed: `{escape_md(res.stderr[:100])}`", parse_mode="MarkdownV2")

# --- UI Callbacks ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "fix_git":
        subprocess.run(f"git remote set-url origin {REPO_URL}", shell=True, cwd=ROOT_DIR)
        await query.edit_message_text("✅ GitHub Connection Repaired\! Try `/sync` now\.", parse_mode="MarkdownV2")
    
    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Quick Run", callback_data=f"qrun_{f}"), 
               InlineKeyboardButton("📄 Logs", callback_data=f"logs_{pid}")],
              [InlineKeyboardButton("🗑 Delete", callback_data=f"cdel_{f}"), 
               InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{f}"
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
        await query.edit_message_text(f"🚀 Started `{escape_md(f)}`\. Use `/logs {f}` to monitor\.", parse_mode="MarkdownV2")

    elif data.startswith("logs_"):
        pid = data.replace("logs_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        try: await query.edit_message_text(f"📄 *Logs:* `{escape_md(pid)}`\n\n{get_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 Process Terminated\.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    
    h = [("start",start_cmd), ("status",status_cmd), ("deployments",deployments_cmd), ("logs",logs_cmd), ("run",run_cmd), ("stop",stop_cmd), ("send",send_cmd), ("upload",upload_cmd), ("sync",sync_cmd)]
    for n, f in h: app.add_handler(CommandHandler(n, f))
    
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
