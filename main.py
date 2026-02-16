import os, subprocess, logging, re, io, time, asyncio, shutil, json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from engine import LabEngine

# --- Configuration ---
engine = LabEngine()
BOT_TOKEN = os.getenv("BOT_TOKEN")
running_processes = {} # { pid: {proc, uid, slug, rollback_hash} }

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- Utilities (Your Original Styles) ---
def escape_md(text): 
    return re.sub(r'([_*\[\]()~`>#+\-=|{}.!])', r'\\\1', str(text))

def format_logs_output(lines):
    if not lines: return r"_Log is empty\._"
    return "\n".join([f"> {escape_md(line.strip())}" for line in lines])

# --- Auto-Deploy Watchdog ---
async def watchdog(context: ContextTypes.DEFAULT_TYPE):
    while True:
        try:
            for uid_folder in os.listdir(engine.root_dir):
                if not uid_folder.isdigit(): continue
                uid = int(uid_folder)
                if engine.git_poll_update(uid):
                    config = engine.read_config(uid)
                    if config and config.get("auto_deploy"):
                        await context.bot.send_message(uid, "🛰 *Auto\-Deploy:* Change detected\. Updating\.\.\.", parse_mode="MarkdownV2")
                        active_pid = next((p for p in running_processes if p.startswith(f"{uid}_")), None)
                        if active_pid: 
                            running_processes[active_pid]['proc'].terminate()
                            del running_processes[active_pid]
                        
                        success, old_hash = engine.deploy_pull(uid)
                        if success:
                            await start_configured_process(uid, config, context, rollback_hash=old_hash)
        except Exception as e: logging.error(f"Watchdog error: {e}")
        await asyncio.sleep(60)

async def start_configured_process(uid, config, context, rollback_hash=None):
    slug = config['name']
    pid = f"{uid}_{slug}"
    exe = engine.get_venv_exe(uid)
    user_path = engine.get_user_base(uid)
    log_p = os.path.join(user_path, f"{pid}.log")
    
    cmd = config['start_cmd'].replace("python3", exe)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.Popen(cmd.split(), cwd=user_path, env=env, stdout=open(log_p, "w"), 
                            stderr=subprocess.STDOUT, stdin=subprocess.PIPE, text=True, bufsize=0)
    
    running_processes[pid] = {"proc": proc, "uid": uid, "slug": slug, "rollback": rollback_hash}
    
    await asyncio.sleep(15)
    if proc.poll() is not None: # Crashed within 15s
        if rollback_hash:
            engine.rollback(uid, rollback_hash)
            await context.bot.send_message(uid, "❌ *Update Failed\!* Rolled back to stable version\.", parse_mode="MarkdownV2")
            await start_configured_process(uid, config, context)
    else:
        await context.bot.send_message(uid, f"✅ *Deployment Stable:* `{escape_md(slug)}`", parse_mode="MarkdownV2")

# --- 1. /start (The Full Introduction) ---
async def start_cmd(update, context):
    uid = update.effective_user.id
    engine.setup_venv(uid)
    msg = (
        r"🤖 *Bot Lab Manager v21\.0*" + "\n"
        r"📦 *ISOLATED GIT & VENV*" + "\n"
        r"• `/clone [url]` — Clone repo to your folder\." + "\n"
        r"• `/sync` — Push to your private branch\." + "\n\n"
        r"📂 *LAB CONTROL*" + "\n"
        r"• `/status` — File explorer \(Private\)\." + "\n"
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

# --- 2. File & Lab Controls ---
async def status_cmd(update, context):
    uid, base = update.effective_user.id, engine.get_user_base(update.effective_user.id)
    all_files = [os.path.relpath(os.path.join(r, f), base) for r, d, files in os.walk(base) if "venv" not in r and ".git" not in r for f in files if not f.endswith(".log")]
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in sorted(all_files)[:15]]
    kb.append([InlineKeyboardButton("🔄 Refresh", callback_data="status_refresh"), InlineKeyboardButton("🏠 Home", callback_data="nav_home")])
    if update.callback_query: await update.callback_query.edit_message_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")
    else: await update.message.reply_text("📂 *Explorer*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def search_cmd(update, context):
    if not context.args: return
    uid, query, base = update.effective_user.id, context.args[0].lower(), engine.get_user_base(update.effective_user.id)
    results = [os.path.relpath(os.path.join(r, f), base) for r, d, files in os.walk(base) if "venv" not in r and ".git" not in r for f in files if query in f.lower()]
    if not results: return await update.message.reply_text("❌ No files found\.")
    kb = [[InlineKeyboardButton(f"📄 {f}", callback_data=f"manage_{f}")] for f in results[:10]]
    await update.message.reply_text(f"🔍 *Results:*", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

async def upload_cmd(update, context):
    uid, base = update.effective_user.id, engine.get_user_base(update.effective_user.id)
    if not update.message.reply_to_message or not context.args: return await update.message.reply_text("❌ Reply with `/upload filename.py`")
    f_name = context.args[0]; target = os.path.join(base, f_name); replied = update.message.reply_to_message
    content = (await (await replied.document.get_file()).download_as_bytearray()).decode('utf-8') if replied.document else replied.text
    with open(target, "w") as f: f.write(content.strip())
    # Manual sync call using logic from your old run_git_push
    await update.message.reply_text(f"✅ Saved locally as `{escape_md(f_name)}`\. Use `/sync` to push to GitHub\.", parse_mode="MarkdownV2")

async def delete_cmd(update, context):
    if not context.args: return
    target = os.path.join(engine.get_user_base(update.effective_user.id), context.args[0])
    if os.path.exists(target): 
        if os.path.isdir(target): shutil.rmtree(target)
        else: os.remove(target)
        await update.message.reply_text("🗑 Deleted\.")

# --- 3. Execution & Git ---
async def run_cmd(update, context):
    uid = update.effective_user.id
    config = engine.read_config(uid)
    if not config:
        template = {"name": "my-bot", "start_cmd": "python3 main.py", "auto_deploy": True}
        return await update.message.reply_text(f"❌ Missing `bot.json`\!\nUpload this template:\n`{escape_md(json.dumps(template))}`", parse_mode="MarkdownV2")
    await start_configured_process(uid, config, context)

async def stop_cmd(update, context):
    if not context.args: return
    uid, slug = update.effective_user.id, context.args[0]
    pid = f"{uid}_{slug}"
    if pid in running_processes:
        running_processes[pid]['proc'].terminate()
        del running_processes[pid]
        await update.message.reply_text(f"🛑 Stopped `{escape_md(slug)}`\.")

async def deployments_cmd(update, context):
    uid, prefix = update.effective_user.id, f"{update.effective_user.id}_"
    procs = [v['slug'] for k, v in running_processes.items() if k.startswith(prefix)]
    msg = "🛰 *Active Tasks:*\n" + "\n".join([f"✅ `{escape_md(p)}`" for p in procs]) if procs else r"📭 No active tasks\."
    await update.message.reply_text(msg, parse_mode="MarkdownV2")

async def sync_cmd(update, context):
    # This calls your GitHub sync logic
    await update.message.reply_text("🔄 Syncing to GitHub branch...")
    # Add your git push logic here if needed, or rely on auto-deploy

async def clone_cmd(update, context):
    if not context.args: return
    uid, url = update.effective_user.id, context.args[0]
    target = engine.get_user_base(uid)
    res = subprocess.run(f"git clone {url} .", shell=True, cwd=target, capture_output=True)
    await update.message.reply_text("✅ Cloned\!" if res.returncode == 0 else "❌ Failed\.", parse_mode="MarkdownV2")

async def handle_callback(update, context):
    query = update.callback_query; uid, data = query.from_user.id, query.data
    await query.answer()
    if data == "status_refresh": await status_cmd(update, context)
    elif data == "view_deploys": await deployments_cmd(update, context)
    elif data == "nav_home": await start_cmd(update, context)

if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    handlers = [
        ("start", start_cmd), ("status", status_cmd), ("search", search_cmd),
        ("upload", upload_cmd), ("delete", delete_cmd), ("run", run_cmd),
        ("stop", stop_cmd), ("deployments", deployments_cmd), ("sync", sync_cmd),
        ("clone", clone_cmd), ("logs", logs_cmd), ("send", send_cmd)
    ]
    for n, f in handlers: app.add_handler(CommandHandler(n, f))
    app.add_handler(CallbackQueryHandler(handle_callback))
    
    asyncio.get_event_loop().create_task(watchdog(app))
    
    print("Bot Lab v21.0 Online")
    app.run_polling()
