import os, shutil, json
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from engine import LabEngine

engine = LabEngine()

async def handle_callback(update, context, running_processes, start_configured_process, get_formatted_logs, start_cmd, status_cmd, deployments_cmd):
    query = update.callback_query
    uid, data = query.from_user.id, query.data
    await query.answer()

    # --- Navigation ---
    if data == "status_refresh": 
        await status_cmd(update, context)
    elif data == "nav_home": 
        await start_cmd(update, context)
    elif data == "view_deploys": 
        await deployments_cmd(update, context)
    
    # --- File Management ---
    elif data.startswith("manage_"):
        f_path = data.replace("manage_", "")
        pid = f"{uid}_{os.path.basename(f_path)}"
        kb = [
            [InlineKeyboardButton("📄 Logs", callback_data=f"logs_{f_path}"), 
             InlineKeyboardButton("🗑 Delete", callback_data=f"del_{f_path}")],
            [InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]
        ]
        await query.edit_message_text(f"📄 *File:* `{f_path}`", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    # --- Dynamic Actions ---
    elif data.startswith("logs_"):
        slug = data.replace("logs_", "")
        logs = get_formatted_logs(uid, slug)
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data=f"logs_{slug}"), InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]
        await query.edit_message_text(f"📄 *Logs:* \n{logs}", reply_markup=InlineKeyboardMarkup(kb), parse_mode="MarkdownV2")

    elif data.startswith("del_"):
        f_path = data.replace("del_", "")
        target = os.path.join(engine.get_user_base(uid), f_path)
        if os.path.exists(target):
            shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
        await query.edit_message_text(f"🗑 Deleted `{f_path}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="status_refresh")]]), parse_mode="MarkdownV2")

    elif data == "quick_run":
        config = engine.read_config(uid)
        if config: await start_configured_process(uid, config, context)
