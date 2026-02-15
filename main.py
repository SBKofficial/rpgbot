import os, subprocess, psutil, time, pathlib, logging, shutil, re, io
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes

# --- Configuration & Safety ---
ROOT_DIR = os.path.abspath("./bot_lab")
pathlib.Path(ROOT_DIR).mkdir(parents=True, exist_ok=True)
running_processes = {}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def escape_md(text): return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))
def get_user_base(uid): return os.path.abspath(os.path.join(ROOT_DIR, str(uid)))
def is_safe(uid, path): return os.path.abspath(path).startswith(get_user_base(uid))

def get_logs(uid, pid):
    path = os.path.join(get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ No logs found\."
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            if not lines: return r"_Log is currently empty\._"
            return "\n".join([f"> {escape_md(line.strip())}" for line in lines])
    except: return r"❌ Error reading logs\."

async def post_init(app):
    # This syncs the Menu button in Telegram
    cmds = [("start","📖 Manual"), ("status","📂 Explorer"), ("deployments","🛰 Tasks"), 
            ("logs","📄 Logs"), ("run","▶️ Run"), ("stop","🛑 Stop"), 
            ("send","⌨️ OTP"), ("upload","📥 Upload")]
    await app.bot.set_my_commands([BotCommand(c, d) for c, d in cmds])

# --- COMMAND: /start (The Detailed Manual) ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        r"🤖 *Bot Lab Manager v14\.8*" + "\n"
        r"\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-" + "\n"
        r"🛰 *Core Navigation*" + "\n"
        r"• `/status` — Open File Explorer to run/delete files\." + "\n"
        r"• `/deployments` — View and manage active processes\." + "\n\n"
        
        r"📂 *File Management*" + "\n"
        r"• `/upload [name]` — Reply to a file/code to save it\. " + "\n"
        r"  _Auto-detects \.py or \.js extensions from original file\._" + "\n\n"
        
        r"▶️ *Process Control*" + "\n"
        r"• `/run [slug] [cmd]` — Start a custom command\." + "\n"
        r"  _Example: `/run nexus node waifu.js`_" + "\n"
        r"• `/stop [slug]` — Kill a running process by its name\." + "\n"
        r"• `/logs [slug]` — Get a snapshot of the latest output\." + "\n\n"
        
        r"⌨️ *Terminal Input*" + "\n"
        r"• `/send [slug] [text]` — Send OTP or text to a process\." + "\n"
        r"  _Example: `/send nexus 12345`_"
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"),
           InlineKeyboardButton("🛰 Active Tasks", callback_data="view_deploys")]]
    
    if update.callback_query: 
        await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: 
        await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

# --- UI & LOGIC HANDLERS ---
async def status_cmd(update, context):
    uid = update.effective_user.id; curr = context.user_data.get("path", get_user_base(uid))
    kb = []
    try:
        for item in sorted(os.listdir(curr)):
            p = os.path.join(curr, item); rel = os.path.relpath(p, get_user_base(uid))
            icon = "📁" if os.path.isdir(p) else "🐍" if item.endswith(".py") else "🟢" if item.endswith(".js") else "📄"
            kb.append([InlineKeyboardButton(f"{icon} {item}", callback_data=f"open_{rel}" if os.path.isdir(p) else f"manage_{rel}")])
    except: pass
    if curr != get_user_base(uid): kb.append([InlineKeyboardButton("⬅️ Back", callback_data="nav_back")])
    kb.append([InlineKeyboardButton("📖 Manual", callback_data="nav_home")])
    text = f"📂 *Explorer:* `{escape_md(os.path.relpath(curr, get_user_base(uid)))}`"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [n.replace(prefix, "") for n in running_processes if n.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No tasks\."
    kb = [[InlineKeyboardButton("📖 Manual", callback_data="nav_home")]]
    if update.callback_query: await update.callback_query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/logs [name]`")
    uid, pid = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}"
    kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh_{pid}"), InlineKeyboardButton("🗑 Clear", callback_data=f"clearlog_{pid}")]]
    await update.message.reply_text(f"📄 *Logs:* `{escape_md(context.args[0])}`\n\n{get_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def run_cmd(update, context):
    if len(context.args) < 2: return await update.message.reply_text("❌ Usage: `/run [slug] [command]`")
    uid, pid, cmd = update.effective_user.id, f"{update.effective_user.id}_{context.args[0]}", " ".join(context.args[1:])
    log_p = os.path.join(get_user_base(uid), f"{pid}.log")
    running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdin=subprocess.PIPE, stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
    await update.message.reply_text(fr"🚀 Running `{escape_md(context.args[0])}`")

async def stop_cmd(update, context):
    if not context.args: return
    pid = f"{update.effective_user.id}_{context.args[0]}"
    if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
    await update.message.reply_text(fr"🛑 Stopped `{escape_md(context.args[0])}`")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    pid, text = f"{update.effective_user.id}_{context.args[0]}", "".join(context.args[1:])
    if pid in running_processes and running_processes[pid].poll() is None:
        running_processes[pid].stdin.write(text + "\n"); running_processes[pid].stdin.flush()
        await update.message.reply_text(fr"⌨️ Sent `{escape_md(text)}` to stdin\.")

async def upload_cmd(update, context):
    uid, base = update.effective_user.id, get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return
    replied = update.message.reply_to_message
    input_name = context.args[0]
    ext = "." + replied.document.file_name.split(".")[-1] if replied.document and "." in replied.document.file_name else ""
    filename = input_name + ext if "." not in input_name and ext else input_name
    target = os.path.join(base, filename)
    raw = (await (await replied.document.get_file()).download_to_memory()).decode('utf-8') if replied.document else replied.text
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n"); clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean.replace("```", "")
    with open(target, "w") as f: f.write(clean)
    await update.message.reply_text(fr"✅ Saved as `{escape_md(filename)}`")

# --- CALLBACK ROUTER ---
async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()

    if data == "nav_home": await start_cmd(update, context)
    elif data == "status_refresh": await status_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "nav_back":
        curr = context.user_data.get("path", get_user_base(uid)); p = os.path.dirname(curr)
        if is_safe(uid, p): context.user_data["path"] = p
        await status_cmd(update, context)
    elif data.startswith("open_"):
        context.user_data["path"] = os.path.join(get_user_base(uid), data.replace("open_", ""))
        await status_cmd(update, context)
    elif data.startswith("manage_"):
        f = data.replace("manage_", ""); kb = [[InlineKeyboardButton("▶️ Run", callback_data=f"qrun_{f}"), InlineKeyboardButton("🗑 Delete", callback_data=f"askdel_{f}")], [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *File:* `{f}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    elif data.startswith("qrun_"):
        f = data.replace("qrun_", ""); pid = f"{uid}_{os.path.basename(f)}"; log_p = os.path.join(get_user_base(uid), f"{pid}.log")
        cmd = f"node {f}" if f.endswith(".js") else f"python3 -u {f}" if f.endswith(".py") or "." not in f else f"bash {f}"
        running_processes[pid] = subprocess.Popen(cmd, shell=True, cwd=get_user_base(uid), stdin=subprocess.PIPE, stdout=open(log_p, "w"), stderr=subprocess.STDOUT, text=True)
        await query.edit_message_text(fr"🚀 Running via `{cmd.split()[0]}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 View Logs", callback_data=f"refresh_{pid}")]]))
    elif data.startswith("refresh_"):
        pid = data.replace("refresh_", "")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh_{pid}"), InlineKeyboardButton("🗑 Clear", callback_data=f"clearlog_{pid}")], [InlineKeyboardButton("🛑 Stop", callback_data=f"stopbutton_{pid}")]]
        await query.edit_message_text(f"📄 *Logs:* `{pid.split('_',1)[1]}`\n\n{get_logs(uid, pid)}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    elif data.startswith("clearlog_"):
        pid = data.replace("clearlog_", ""); log_p = os.path.join(get_user_base(uid), f"{pid}.log"); open(log_p, 'w').close() if os.path.exists(log_p) else None
        await query.edit_message_text(r"🗑 *Log cleared\.*", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data=f"refresh_{pid}")]]), parse_mode="MarkdownV2")
    elif data.startswith("stopbutton_"):
        pid = data.replace("stopbutton_", "")
        if pid in running_processes: running_processes[pid].terminate(); del running_processes[pid]
        await query.edit_message_text(r"🛑 *Terminated\.*", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="view_deploys")]]), parse_mode="MarkdownV2")
    elif data.startswith("askdel_"):
        f = data.replace("askdel_", ""); kb = [[InlineKeyboardButton("✅ Yes", callback_data=f"cdel_{f}"), InlineKeyboardButton("❌ No", callback_data=f"manage_{f}")]]
        await query.edit_message_text(fr"🗑 Delete `{f}`?", reply_markup=InlineKeyboardMarkup(kb))
    elif data.startswith("cdel_"):
        f = data.replace("cdel_", ""); p = os.path.join(get_user_base(uid), f)
        if is_safe(uid, p): (os.remove(p) if os.path.isfile(p) else shutil.rmtree(p))
        await query.edit_message_text(f"🗑 Deleted `{f}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]))

# --- Main Boot ---
if __name__ == '__main__':
    TOKEN = "8341690614:AAFsv3JbPeD98gScm23lqiuobcjSLWCfMVA"
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    
    # REGISTERING ALL 8 HANDLERS
    handlers = [("start",start_cmd), ("status",status_cmd), ("deployments",deployments_cmd), 
                ("logs",logs_cmd), ("run",run_cmd), ("stop",stop_cmd), ("send",send_cmd), ("upload",upload_cmd)]
    for c, f in handlers: app.add_handler(CommandHandler(c, f))
    
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.run_polling()
