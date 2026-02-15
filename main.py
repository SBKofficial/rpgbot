import os, subprocess, psutil, time, pathlib, logging, shutil, re, io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import telegram.error

# --- Secure Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

GIT_TOKEN = os.getenv("GIT_TOKEN")
REPO_URL = f"https://{GIT_TOKEN}@github.com/SBKofficial/rpgbot.git" if GIT_TOKEN else None

running_processes = {} 
logging.basicConfig(level=logging.INFO)

# --- Utilities ---
def escape_md(text): 
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_user_base(uid): 
    path = os.path.join(LAB_DIR, str(uid))
    os.makedirs(path, exist_ok=True)
    return path

def get_formatted_logs(uid, pid):
    """Fetches logs and formats them into a Blockquote with '>'"""
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ _No logs available yet\._"
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            if not lines: return r"_Log is currently empty\._"
            content = "\n".join([f"> {escape_md(line.strip())}" for line in lines])
            return content
    except: return r"❌ _Error reading logs\._"

# --- 1. /start (Complete Introduction) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v17\.0*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"📂 *FILE COMMANDS*" + "\n"
        r"• `/status` — Explorer to run/delete files\." + "\n"
        r"• `/upload [name]` — Save code by replying to a file\." + "\n"
        r"• `/sync` — Push all files to GitHub\." + "\n\n"
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — View active processes\." + "\n"
        r"• `/logs [name]` — View output in blockquote format\." + "\n\n"
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [slug] [cmd]` — Start a process manually\." + "\n"
        r"• `/stop [slug]` — Kill a process\." + "\n"
        r"• `/send [slug] [text]` — Send text to a bot's stdin\." + "\n\n"
        r"🔧 *GIT REPAIR*" + "\n"
        r"Use the button below to fix 403 Forbidden errors\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")],
          [InlineKeyboardButton("🔧 Repair GitHub", callback_data="fix_git")]]
    
    if update.callback_query:
        try: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status & 3. /deployments ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    base = get_user_base(uid)
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in sorted(os.listdir(base)) if not f.endswith(".log") and f != ".git"]
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

# --- 4. /upload ---
async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return
    filename, replied = context.args[0], update.message.reply_to_message
    target = os.path.join(base, filename)
    
    if replied.document:
        f_obj = await replied.document.get_file()
        out = io.BytesIO()
        await f_obj.download_to_memory(out=out)
        raw = out.getvalue().decode('utf-8')
    else: raw = replied.text

    with open(target, "w") as f: f.write(raw.strip())
    m = await update.message.reply_text("💾 Saved locally. Syncing...")
    
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run(f"git commit -m 'Upload {filename}'", shell=True, cwd=ROOT_DIR)
    res = subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    
    if res.returncode == 0: await m.edit_text(f"✅ `{escape_md(filename)}` pushed to GitHub!")
    else: await m.edit_text(f"⚠️ Saved locally, but Push failed. Use Repair button.")

# --- 5-9. Process Commands ---
async def run_cmd(update, context):
    if len(context.args) < 2: return
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
    await update.message.reply_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid, text = f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
    if pid in running_processes:
        running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush()
        await update.message.reply_text(f"⌨️ Sent to `{escape_md(context.args[0])}`")

async def sync_cmd(update, context):
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run("git commit -m 'Sync'", shell=True, cwd=ROOT_DIR)
    res = subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    if res.returncode == 0: await update.message.reply_text("✅ GitHub Sync OK")
    else: await update.message.reply_text(f"❌ Failed: {escape_md(res.stderr[:50])}")

# --- UI Callback Logic ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "fix_git":
        if REPO_URL:
            subprocess.run(f"git remote set-url origin {REPO_URL}", shell=True, cwd=ROOT_DIR)
            await query.edit_message_text("✅ Git Connection Repaired!")
        else: await query.edit_message_text("❌ GIT_TOKEN variable missing!")
    
    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Quick Run", callback_data=f"qrun_{f}"), 
               InlineKeyboardButton("📄 View Logs", callback_data=f"showlogs_{pid}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{f}"
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
        # FLIP TO LOGS IMMEDIATELY
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"🚀 *Running:* `{escape_md(f)}` \n\n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        try: await query.edit_message_text(f"📄 *Logs:* \n{get_formatted_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes:
            running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 Process Terminated.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    
    # 9 Full Commands Registered
    cmds_list = [("start", start_cmd), ("upload", upload_cmd), ("status", status_cmd), 
                 ("deployments", deployments_cmd), ("run", run_cmd), ("stop", stop_cmd), 
                 ("logs", logs_cmd), ("send", send_cmd), ("sync", sync_cmd)]
    
    for n, f in cmds_list: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
