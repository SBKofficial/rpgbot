import os, subprocess, psutil, time, pathlib, logging, shutil, re, io
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration ---
ROOT_DIR = os.path.abspath(".") 
pathlib.Path(os.path.join(ROOT_DIR, "bot_lab")).mkdir(parents=True, exist_ok=True)
running_processes = {} 

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- Utilities ---
def escape_md(text): 
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_user_base(uid): 
    return os.path.abspath(os.path.join(ROOT_DIR, "bot_lab", str(uid)))

def get_logs(uid, pid):
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ No logs found\."
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            return "\n".join([f"`{escape_md(line.strip())}`" for line in lines]) if lines else r"_Log is empty\._"
    except: return r"❌ Error reading logs\."

# --- GitHub Persistence ---
def sync_to_github(user_id, filename, action="Sync"):
    try:
        subprocess.run("git config user.email 'bot@lab.com'", shell=True, cwd=ROOT_DIR)
        subprocess.run("git config user.name 'BotLabManager'", shell=True, cwd=ROOT_DIR)
        subprocess.run("git add .", shell=True, cwd=ROOT_DIR)
        subprocess.run(f"git commit -m '{action}: {filename} for {user_id}'", shell=True, cwd=ROOT_DIR)
        subprocess.run("git push", shell=True, cwd=ROOT_DIR)
        return True
    except: return False

# --- 1. /start (Full Manual) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v15\.9*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"📂 *Files:* `/status` \| `/upload [name]` \| `/sync`" + "\n"
        r"🛰 *Monitor:* `/deployments` \| `/logs [slug]`" + "\n"
        r"▶️ *Control:* `/run [slug] [cmd]` \| `/stop [slug]` \| `/send [slug] [text]`" + "\n\n"
        r"✨ *UI Guide:* Use `/status` to browse files and click *Run* to start bots with auto-log tracking\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")],
          [InlineKeyboardButton("🔄 Full GitHub Sync", callback_data="manual_sync")]]
    
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 2. /status (Explorer) ---
async def status_cmd(update, context):
    uid = update.effective_user.id
    curr = context.user_data.get("path", get_user_base(uid))
    kb = []
    if os.path.exists(curr):
        for item in sorted(os.listdir(curr)):
            if item.endswith(".log") or item == ".git": continue
            p = os.path.join(curr, item)
            rel = os.path.relpath(p, get_user_base(uid))
            icon = "📁" if os.path.isdir(p) else "🐍" if item.endswith(".py") else "🟢" if item.endswith(".js") else "📄"
            kb.append([InlineKeyboardButton(f"{icon} {item}", callback_data=f"manage_{rel}")])
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("📖 Home", callback_data="nav_home")])
    text = f"📂 *Explorer:* `{escape_md(os.path.basename(curr))}`"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 3. /deployments ---
async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="view_deploys"), InlineKeyboardButton("📖 Home", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- 4. /upload ---
async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return
    os.makedirs(base, exist_ok=True)
    replied = update.message.reply_to_message
    input_name = context.args[0]
    ext = "." + replied.document.file_name.split(".")[-1] if replied.document else ""
    filename = input_name + ext if "." not in input_name else input_name
    target = os.path.join(base, filename)
    
    if replied.document:
        file = await replied.document.get_file()
        out = io.BytesIO(); await file.download_to_memory(out=out)
        raw = out.getvalue().decode('utf-8')
    else: raw = replied.text

    with open(target, "w") as f: f.write(raw.strip())
    m = await update.message.reply_text("💾 Saved... Syncing...")
    if sync_to_github(uid, filename, "Upload"): await m.edit_text(f"✅ `{escape_md(filename)}` synced to GitHub!")

# --- 5, 6, 7, 8, 9. Run, Stop, Logs, Send, Sync ---
async def run_cmd(update, context):
    if len(context.args) < 2: return
    uid, pid, cmd = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
    log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
    await update.message.reply_text(f"🚀 Started `{context.args[0]}`")

async def stop_cmd(update, context):
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes:
        running_processes[pid].terminate(); del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{context.args[0]}`")

async def logs_cmd(update, context):
    if not context.args: return
    uid, pid = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}"
    await update.message.reply_text(f"📄 *Logs:* {get_logs(uid, pid)}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid, text = f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
    if pid in running_processes:
        running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush()
        await update.message.reply_text(f"⌨️ Sent to `{context.args[0]}`")

async def sync_cmd(update, context):
    if sync_to_github(update.effective_user.id, "Manual", "Cmd"): await update.message.reply_text("✅ GitHub Sync OK")

# --- UI Logic ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "manual_sync": 
        if sync_to_github(uid, "All", "Btn"): await query.edit_message_text("✅ Sync Complete!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="nav_home")]]))

    elif data.startswith("manage_"):
        f = data.replace("manage_", "")
        kb = [[InlineKeyboardButton("▶️ Quick Run", callback_data=f"qrun_{f}"), InlineKeyboardButton("🗑 Delete", callback_data=f"askdel_{f}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *File:* `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    
    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{os.path.basename(f)}"
        log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}"
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdout=open(log_p, "w"), stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True)
        kb = [[InlineKeyboardButton("📄 View Logs", callback_data=f"logs_{pid}")]]
        await query.edit_message_text(f"🚀 Running `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("logs_"):
        pid = data.replace("logs_", ""); name = pid.split("_", 1)[1]
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logs_{pid}"), InlineKeyboardButton("🛑 Stop", callback_data=f"kill_{pid}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *Logs:* `{escape_md(name)}`\n\n{get_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("kill_"):
        pid = data.replace("kill_", "")
        if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text("🛑 Terminated.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Explorer", callback_data="status_refresh")]]))

    elif data.startswith("askdel_"):
        f = data.replace("askdel_", ""); kb = [[InlineKeyboardButton("✅ Delete", callback_data=f"cdel_{f}")], [InlineKeyboardButton("❌ Cancel", callback_data=f"manage_{f}")]]
        await query.edit_message_text(f"🗑 Delete `{escape_md(f)}`?", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("cdel_"):
        f = data.replace("cdel_", ""); p = os.path.join(get_user_base(uid), f)
        if os.path.exists(p): os.remove(p)
        await query.edit_message_text(f"🗑 Deleted `{escape_md(f)}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

if __name__ == '__main__':
    TOKEN = os.getenv("BOT_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()
    cmds = [("start",start_cmd), ("status",status_cmd), ("deployments",deployments_cmd), ("logs",logs_cmd), ("run",run_cmd), ("stop",stop_cmd), ("send",send_cmd), ("upload",upload_cmd), ("sync",sync_cmd)]
    for n, f in cmds: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
