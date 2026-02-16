import os, subprocess, logging, re, asyncio, shutil, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from engine import LabEngine

# --- Initialization ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, slug, uid} }

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def escape_md(text):
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def get_formatted_logs(uid, slug):
    pid = f"{uid}_{slug}"
    path = os.path.join(engine.get_user_base(uid), f"{pid}.log")
    if not os.path.exists(path): return r"⚠️ _No logs available\._"
    try:
        with open(path, "r") as f:
            lines = f.readlines()[-15:]
            return "\n".join([f"> {escape_md(l.strip())}" for l in lines]) if lines else r"_Log is empty\._"
    except: return r"❌ _Error reading logs\._"

# --- Watchdog for Auto-Deploy ---
async def watchdog(context: ContextTypes.DEFAULT_TYPE):
    while True:
        for uid_folder in os.listdir(engine.root_dir):
            if not uid_folder.isdigit(): continue
            uid = int(uid_folder)
            if engine.git_poll_update(uid):
                config = engine.read_config(uid)
                if config and config.get("auto_deploy"):
                    await context.bot.send_message(uid, "🛰 *Auto\-Deploy:* Update detected\. Pulling\.\.\.", parse_mode="MarkdownV2")
                    # Stop existing
                    pid = next((p for p in running_processes if p.startswith(f"{uid}_")), None)
                    if pid:
                        running_processes[pid]['proc'].terminate()
                        del running_processes[pid]
                    success, old_hash = engine.deploy_pull(uid)
                    if success: await start_configured_process(uid, config, context)
        await asyncio.sleep(60)

async def start_configured_process(uid, config, context):
    slug = config['name']
    pid = f"{uid}_{slug}"
    exe = engine.get_venv_exe(uid)
    user_path = engine.get_user_base(uid)
    log_p = os.path.join(user_path, f"{pid}.log")
    
    cmd = config['start_cmd'].replace("python3", exe)
    proc = subprocess.Popen(cmd, shell=True, cwd=user_path, env={"PYTHONUNBUFFERED":"1"}, 
                            stdout=open(log_p, "w"), stderr=subprocess.STDOUT, 
                            stdin=subprocess.PIPE, text=True, bufsize=0)
    
    running_processes[pid] = {"proc": proc, "slug": slug, "uid": uid}
    await context.bot.send_message(uid, f"🚀 *Started:* `{escape_md(slug)}`", parse_mode="MarkdownV2")

# --- Commands ---
async def start_cmd(update, context):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v21\.0*" + "\n"
        r"📦 *ISOLATED GIT & VENV*" + "\n"
        r"• `/connect [url]` — Link GitHub repository\." + "\n"
        r"• `/sync` — Push changes to your branch\." + "\n"
        r"• `/auto on|off` — Toggle background updates\." + "\n\n"
        r"📂 *LAB CONTROL*" + "\n"
        r"• `/status` — File explorer \(Buttons\)\." + "\n"
        r"• `/upload [name]` — Save code from reply\." + "\n"
        r"• `/search [query]` — Recursive file search\." + "\n"
        r"• `/delete [name]` — Remove file/folder\." + "\n\n"
        r"▶️ *EXECUTION*" + "\n"
        r"• `/run` — Start via `bot.json` config\." + "\n"
        r"• `/send [slug] [text]` — Input for OTPs\." + "\n"
        r"• `/deployments` — List active tasks\." + "\n"
        r"• `/logs [slug]` — View process output\." + "\n"
        r"• `/stop [slug]` — Kill a task\."
    )
    kb = [[InlineKeyboardButton("📂 Explorer", callback_data="status_refresh"), InlineKeyboardButton("🛰 Tasks", callback_data="view_deploys")]]
    await update.message.reply_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def connect_cmd(update, context):
    if not context.args: return await update.message.reply_text("❌ Usage: `/connect [url]`")
    uid = update.effective_user.id
    success, detail = engine.connect_repo(uid, context.args[0])
    if success:
        await update.message.reply_text(f"✅ *Connected\!* Branch: `{detail}`\. Use `/run` to start\.", parse_mode="MarkdownV2")
    else:
        await update.message.reply_text(f"❌ *Failed:* `{escape_md(detail)}`", parse_mode="MarkdownV2")

async def run_cmd(update, context):
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config:
        template = json.dumps(engine.get_config_template(), indent=2)
        return await update.message.reply_text(f"⚠️ *Missing bot\.json\!* Upload this template:\n```json\n{template}\n```", parse_mode="MarkdownV2")
    await start_configured_process(uid, config, context)

async def upload_cmd(update, context):
    if not update.message.reply_to_message or not context.args: return await update.message.reply_text("❌ Reply with `/upload filename.py`")
    uid, f_name = update.effective_user.id, context.args[0]
    target = os.path.join(engine.get_user_base(uid), f_name)
    replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()).decode('utf-8') if replied.document else replied.text
    with open(target, "w") as f: f.write(content.strip())
    await update.message.reply_text(f"✅ Saved `{f_name}` locally\. Use `/sync` to push\.")

async def sync_cmd(update, context):
    uid = update.effective_user.id
    if engine.git_push(uid, "Manual Sync from Bot"):
        await update.message.reply_text("✅ Sync Complete \(Push to Branch\)\!")
    else:
        await update.message.reply_text("❌ Sync Failed\. Check `/connect` status\.")

async def auto_cmd(update, context):
    if not context.args: return await update.message.reply_text("Usage: `/auto on` or `/auto off`")
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config: return await update.message.reply_text("❌ No `bot.json` found\.")
    config["auto_deploy"] = (context.args[0].lower() == "on")
    engine.save_config(uid, config)
    await update.message.reply_text(f"✅ Auto\-Deploy is now *{context.args[0].upper()}*", parse_mode="MarkdownV2")

async def status_cmd(update, context):
    uid, base = update.effective_user.id, engine.get_user_base(update.effective_user.id)
    files = [os.path.relpath(os.path.join(r, f), base) for r, d, fs in os.walk(base) if "venv" not in r and ".git" not in r for f in fs if not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in sorted(files)[:15]]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    if update.callback_query: await update.callback_query.edit_message_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def search_cmd(update, context):
    if not context.args: return
    uid, query, base = update.effective_user.id, context.args[0].lower(), engine.get_user_base(update.effective_user.id)
    results = [os.path.relpath(os.path.join(r, f), base) for r, d, fs in os.walk(base) if "venv" not in r and ".git" not in r for f in fs if query in f.lower()]
    await update.message.reply_text("🔍 *Results:*\n" + "\n".join(results[:10]) if results else "❌ No files found\.", parse_mode="MarkdownV2")

async def stop_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    pid = f"{uid}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{slug}`")

async def deployments_cmd(update, context):
    uid_prefix = f"{update.effective_user.id}_"
    active = [v['slug'] for k, v in running_processes.items() if k.startswith(uid_prefix)]
    await update.message.reply_text("🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in active]) if active else "📭 No active tasks\.", parse_mode="MarkdownV2")

async def logs_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    await update.message.reply_text(f"📄 *Logs for {slug}:*\n{get_formatted_logs(uid, slug)}", parse_mode="MarkdownV2")

async def send_cmd(update, context):
    if len(context.args) < 2: return
    uid, slug, text = update.effective_user.id, context.args[0], " ".join(context.args[1:])
    pid = f"{uid}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].stdin.write(text + "\n")
        running_processes[pid]['proc'].stdin.flush()
        await update.message.reply_text("⌨️ Sent to stdin\.")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(engine.get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target):
        if os.path.isdir(target): shutil.rmtree(target)
        else: os.remove(target)
        await update.message.reply_text("🗑 Deleted\.")

async def cb_handler(update, context):
    query = update.callback_query; data = query.data; uid = query.from_user.id
    await query.answer()
    if data == "status_refresh": await status_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    handlers = [
        ("start", start_cmd), ("connect", connect_cmd), ("run", run_cmd), ("upload", upload_cmd),
        ("sync", sync_cmd), ("auto", auto_cmd), ("status", status_cmd), ("search", search_cmd),
        ("stop", stop_cmd), ("deployments", deployments_cmd), ("logs", logs_cmd), ("send", send_cmd),
        ("delete", delete_cmd)
    ]
    for n, f in handlers: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(cb_handler))
    asyncio.get_event_loop().create_task(watchdog(app))
    app.run_polling()
