import psutil
import time
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Use the escape helper from your main or define it here
import html
def esc(text): return html.escape(str(text))

ADMIN_ID = 7708811819 # Replace with your actual ID

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE, running_processes, engine):
    """Checks system health and subprocess resource usage."""
    uid = update.effective_user.id
    
    if uid != ADMIN_ID:
        return await update.message.reply_text("⛔ <b>Access Denied.</b>", parse_mode="HTML")

    # System-wide metrics
    cpu_usage = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    uptime = time.time() - psutil.boot_time()

    stats_msg = (
        "🖥 <b>PRO-SERVER MONITOR</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🟢 <b>CPU Load:</b> <code>{cpu_usage}%</code>\n"
        f"🧠 <b>System RAM:</b> <code>{ram.percent}%</code>\n"
        f"🕒 <b>Uptime:</b> <code>{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m</code>\n\n"
        "🛰 <b>ACTIVE SUBPROCESSES</b>\n"
    )

    if not running_processes:
        stats_msg += "<i>No bots currently running.</i>"
    else:
        for pid_key, data in running_processes.items():
            proc = data['proc']
            slug = data['slug']
            try:
                p = psutil.Process(proc.pid)
                mem_mb = p.memory_info().rss / (1024**2)
                p_cpu = p.cpu_percent(interval=0.1)
                
                stats_msg += (
                    f"🔸 <b>{esc(slug)}</b>\n"
                    f"   └ RAM: <code>{mem_mb:.1f}MB</code> | CPU: <code>{p_cpu}%</code>\n"
                )
            except:
                stats_msg += f"❌ <b>{esc(slug)}</b>: <i>Process unresponsive.</i>\n"

    await update.message.reply_text(stats_msg, parse_mode="HTML")

async def emergency_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kills every single user process running on the VPS."""
    if update.effective_user.id != ADMIN_ID: return
    
    # This Linux command kills all processes started by the bot's children
    os.system("pkill -u $(whoami) -f 'python3|node|bash'") 
    await update.message.reply_text("☢️ <b>EMERGENCY STOP EXECUTED</b>\nAll user processes have been terminated.")

