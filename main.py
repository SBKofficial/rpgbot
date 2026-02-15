import os, subprocess, psutil, time, pathlib, logging, shutil, re, io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
import telegram.error

# --- Secure Configuration ---
ROOT_DIR = os.path.abspath(".") 
LAB_DIR = os.path.join(ROOT_DIR, "bot_lab")
os.makedirs(LAB_DIR, exist_ok=True)

# Using Environment Variables as you requested
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
    """Fetches logs and formats them into a Blockquote as requested"""
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ _No logs available yet\._"
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:] # Last 15 lines
            if not lines: return r"_Log is currently empty\._"
            # The '>' creates the blockquote effect in MarkdownV2
            content = "\n".join([f"> {escape_md(line.strip())}" for line in lines])
            return content
    except: return r"❌ _Error reading logs\._"

# --- 1. /start (Explaining ALL commands as requested) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v16\.9 — Complete Guide*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"📂 *FILE COMMANDS*" + "\n"
        r"• `/status` — Open Explorer to manage, run, or delete files\." + "\n"
        r"• `/upload [name]` — Save code by replying to a message\." + "\n"
        r"• `/sync` — Manually push all files to GitHub\." + "\n\n"
        r"🛰 *MONITORING*" + "\n"
        r"• `/deployments` — View all active bot processes\." + "\n"
        r"• `/logs [slug]` — View terminal output in blockquote format\." + "\n\n"
        r"▶️ *PROCESS CONTROL*" + "\n"
        r"• `/run [slug] [cmd]` — Start a process manually\." + "\n"
        r"• `/stop [slug]` — Kill a running process\." + "\n"
        r"• `/send [slug] [text]` — Send input to a bot's stdin\." + "\n\n"
        r"🔧 *REPAIR*" + "\n"
        r"Use the button below if GitHub pushes fail with a 403 error\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")],
          [InlineKeyboardButton("🔧 Repair GitHub", callback_data="fix_git")]]
    
    if update.callback_query:
        try: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except: pass
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- UI Callback Logic ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": 
        # Explorer Logic
        base = get_user_base(uid)
        kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in sorted(os.listdir(base)) if not f.endswith(".log")]
        kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
        await query.edit_message_text(f"📂 *Explorer:* `{escape_md(os.path.basename(base))}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); pid = f"{uid}_{f}"
        kb = [[InlineKeyboardButton("▶️ Quick Run", callback_data=f"qrun_{f}"), 
               InlineKeyboardButton("📄 View Logs", callback_data=f"showlogs_{pid}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    # --- Quick Run: Immediately flips to blockquote logs ---
    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{f}"
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), 
                                                 stdout=open(log_p, "w"), stderr=subprocess.STDOUT, 
                                                 stdin=subprocess.PIPE, text=True)
        
        # Immediate UI transition to Logs with Stop/Refresh
        kb = [[InlineKeyboardButton("🔄 Refresh Logs", callback_data=f"showlogs_{pid}"), 
               InlineKeyboardButton("🛑 Stop Process", callback_data=f"kill_{pid}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        
        await query.edit_message_text(f"🚀 *Running:* `{escape_md(f)}`\n\n{get_formatted_logs(uid, pid)}", 
                                     reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    # --- Log Viewer: Blockquote + Refresh + Stop ---
    elif data.startswith("showlogs_"):
        pid = data.replace("showlogs_", ""); name = pid.replace(f"{uid}_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"showlogs_{pid}"), 
               InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")],
              [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        
        try:
            await query.edit_message_text(f"📄 *Logs:* `{escape_md(name)}`\n\n{get_formatted_logs(uid, pid)}", 
                                         reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
        except telegram.error.BadRequest: pass 

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes:
            running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 *Process Terminated\.*", 
                                     reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), 
                                     parse_mode="MarkdownV2")

# --- Command Handler Logic ---
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
    m = await update.message.reply_text("💾 Saved. Syncing...")
    
    # Git Push Logic
    subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
    subprocess.run(f"git commit -m 'Upload {filename}'", shell=True, cwd=ROOT_DIR)
    res = subprocess.run(f"git push {REPO_URL} main", shell=True, capture_output=True, text=True, cwd=ROOT_DIR)
    
    if res.returncode == 0: await m.edit_text(f"✅ `{escape_md(filename)}` pushed to GitHub\!")
    else: await m.edit_text(f"⚠️ Saved locally, but Push failed \(403\)\. Use Repair button in /start\.")

# (Include deployments_cmd, status_cmd, run_cmd, stop_cmd, logs_cmd, send_cmd, sync_cmd here)

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    
    cmds = [("start", start_cmd), ("upload", upload_cmd), ("status", status_cmd), 
            ("deployments", deployments_cmd), ("run", run_cmd), ("stop", stop_cmd), 
            ("logs", logs_cmd), ("send", send_cmd), ("sync", sync_cmd)]
    
    for n, f in cmds: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
