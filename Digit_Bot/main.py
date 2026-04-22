import asyncio
import logging
import random
import time
import aiosqlite
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

# --- CONFIGURATION ---
BOT_TOKEN = "8201536495:AAFmMAqszMoGBD2L_LX1AaWYpf1dg974gBo"
DB_NAME = "arena_players.db"

# --- GLOBALS & STATE ---
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Format: {chat_id: {"type": "hc"|"zo", "host": uid, "msg_id": 123, "last_action": timestamp, ...}}
active_games = {}

# --- UI TEMPLATES ---
LOBBY_UI = """
<blockquote><b>『 𓄀 𝔸ℝ𝔼ℕ𝔸 𝕃𝕆𝔹𝔹𝕐 𓄀 』</b>

Welcome, <a href="tg://user?id={user_id}">{name}</a>. 

I am the Arena Bot. I run quick, no-nonsense number games for your group chats. No heavy setups or confusing rules—just add me to a chat and start playing.</blockquote>

<i>Select a guide below to see how the games work, or add me to your crew.</i>
"""

GUIDES_MAIN_UI = """
<blockquote><b>『 𝔾𝔸𝕄𝔼 𝕃𝕀𝔹ℝ𝔸ℝ𝕐 』</b>

Select a game below to view its rules and commands.</blockquote>
"""

GUIDE_HC_UI = """
<blockquote><b>🎯 ℍ𝕆𝕋 & ℂ𝕆𝕃𝔻</b>

<b>𝔾𝕆𝔸𝕃:</b> Find the secret number by spamming guesses in the chat!

<b>ℂ𝕆𝕄𝕄𝔸ℕ𝔻:</b> <code>/nguess [range]</code> (e.g., /nguess 1000)

<b>ℝ𝕌𝕃𝔼𝕊:</b>
• The bot provides a Live Radar with 🔼/🔽 hints.
• First person to type the exact number wins the round.
• 10-minute inactivity auto-close.
• Host can use <code>/endgame</code> to cancel.</blockquote>
"""

GUIDE_ZO_UI = """
<blockquote><b>📉 ℤ𝔼ℝ𝕆 𝕆𝕌𝕋</b>

<b>𝔾𝕆𝔸𝕃:</b> A strategic countdown. Don't be the one to hit zero!

<b>ℂ𝕆𝕄𝕄𝔸ℕ𝔻:</b> <code>/zeroout [number]</code> (e.g., /zeroout 50)

<b>ℝ𝕌𝕃𝔼𝕊:</b>
• Subtract 1, 2, 3, or 4 using inline buttons.
• Players cannot make two moves in a row.
• 10-minute inactivity auto-close.
• Host can use <code>/endgame</code> to cancel.</blockquote>
"""

# --- HELPER FUNCTIONS ---
async def init_db():
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def register_user(user_id: int):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
        await db.commit()

async def is_registered(user_id: int) -> bool:
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)) as cursor:
            return await cursor.fetchone() is not None

def get_player_name(user: types.User) -> str:
    if user.id == 7708811819:
        return "『 𓄀 𝙔𝙖𝙢𝙞 𝙎𝙪𝙠𝙞𝙝𝙚𝙧𝙤 ✗ 』"
    return user.first_name

async def can_bot_pin(chat_id: int) -> bool:
    try:
        member = await bot.get_chat_member(chat_id, bot.id)
        if member.status == "administrator" and member.can_pin_messages:
            return True
    except Exception:
        pass
    return False

# --- KEYBOARDS ---
def get_home_kb(bot_username: str):
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Game Guides", callback_data="help_main")],
        [InlineKeyboardButton(text="➕ Add to Group", url=f"https://t.me/{bot_username}?startgroup=true")]
    ])

def get_guides_main_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎯 Hot & Cold", callback_data="help_hc"),
         InlineKeyboardButton(text="📉 Zero Out", callback_data="help_zo")],
        [InlineKeyboardButton(text="🏠 Home", callback_data="home")]
    ])

def get_back_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Back", callback_data="help_main"),
         InlineKeyboardButton(text="🏠 Home", callback_data="home")]
    ])

def get_zeroout_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="-1", callback_data="zo_1"),
            InlineKeyboardButton(text="-2", callback_data="zo_2"),
            InlineKeyboardButton(text="-3", callback_data="zo_3"),
            InlineKeyboardButton(text="-4", callback_data="zo_4")
        ]
    ])

# --- BACKGROUND SWEEPER ---
async def inactivity_sweeper():
    """Checks for games inactive for > 10 mins and closes them."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        expired_chats = []
        for chat_id, game in active_games.items():
            if now - game["last_action"] > 600: # 10 mins
                expired_chats.append((chat_id, game["msg_id"]))
                
        for chat_id, msg_id in expired_chats:
            try:
                await bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text="🛑 <b>𝕄𝔸𝕋ℂℍ 𝔸𝔹𝕆ℝ𝕋𝔼𝔻</b>\n\nArena closed due to 10 minutes of inactivity."
                )
                await bot.unpin_chat_message(chat_id=chat_id, message_id=msg_id)
            except Exception:
                pass
            if chat_id in active_games:
                del active_games[chat_id]

# --- COMMAND HANDLERS ---
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    if message.chat.type in ["group", "supergroup"]:
        me = await bot.get_me()
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="🤖 Register Here", url=f"https://t.me/{me.username}?start=register")]
        ])
        await message.answer("🎯 <b>𝔸ℝ𝔼ℕ𝔸 𝔹𝕆𝕋 𝕆ℕ𝕃𝕀ℕ𝔼</b>\n\n<i>All players must be registered in my DMs to trigger commands.</i>", reply_markup=kb)
        return

    await register_user(message.from_user.id)
    me = await bot.get_me()
    name = get_player_name(message.from_user)
    await message.answer(LOBBY_UI.format(user_id=message.from_user.id, name=name), reply_markup=get_home_kb(me.username))

@dp.callback_query(F.data.in_(["home", "help_main", "help_hc", "help_zo"]))
async def handle_help_menus(callback: types.CallbackQuery):
    me = await bot.get_me()
    if callback.data == "home":
        text = LOBBY_UI.format(user_id=callback.from_user.id, name=get_player_name(callback.from_user))
        kb = get_home_kb(me.username)
    elif callback.data == "help_main":
        text = GUIDES_MAIN_UI
        kb = get_guides_main_kb()
    elif callback.data == "help_hc":
        text = GUIDE_HC_UI
        kb = get_back_kb()
    else:
        text = GUIDE_ZO_UI
        kb = get_back_kb()

    try:
        await callback.message.edit_text(text, reply_markup=kb)
    except TelegramBadRequest:
        pass
    await callback.answer()

@dp.message(Command("endgame"))
async def cmd_endgame(message: types.Message):
    if message.chat.id not in active_games:
        return
    
    game = active_games[message.chat.id]
    if message.from_user.id != game["host"]:
        return await message.reply("⚠️ Only the host can abort this match.")

    try:
        await bot.edit_message_text(
            chat_id=message.chat.id, message_id=game["msg_id"],
            text="🛑 <b>𝕄𝔸𝕋ℂℍ 𝔸𝔹𝕆ℝ𝕋𝔼𝔻</b>\n\nThe host forcefully terminated the arena."
        )
        await bot.unpin_chat_message(chat_id=message.chat.id, message_id=game["msg_id"])
    except Exception:
        pass

    del active_games[message.chat.id]
    await message.delete()

@dp.message(Command("nguess"))
async def cmd_nguess(message: types.Message):
    if not await is_registered(message.from_user.id):
        return await message.reply("⚠️ Please register in my DMs first to start games.")
    
    if message.chat.id in active_games:
        return await message.reply("🛑 An arena match is already active here!")

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        return await message.reply("❌ <b>Invalid Usage:</b>\n▶️ <code>/nguess 1000</code> (Starts a 1-1000 range)")
    
    max_range = int(args[1])
    target = random.randint(1, max_range)
    has_pin_rights = await can_bot_pin(message.chat.id)
    
    ui_text = f"""🎯 <b>ℍ𝕆𝕋 & ℂ𝕆𝕃𝔻 𝔸ℝ𝔼ℕ𝔸</b>
Target Range: <code>1 - {max_range}</code>
Host: <a href="tg://user?id={message.from_user.id}">{get_player_name(message.from_user)}</a>

<i>Waiting for guesses...</i>
"""
    if not has_pin_rights:
        ui_text += "\n\n<i>⚠️ Note: I am not an Admin. Game board will not be pinned.</i>"

    sent_msg = await message.answer(ui_text)
    
    if has_pin_rights:
        try:
            await bot.pin_chat_message(message.chat.id, sent_msg.message_id)
        except Exception:
            pass

    active_games[message.chat.id] = {
        "type": "hc",
        "host": message.from_user.id,
        "msg_id": sent_msg.message_id,
        "target": target,
        "max": max_range,
        "history": [],
        "last_action": time.time(),
        "last_edit": time.time(),
        "latest_guess": None
    }

@dp.message(Command("zeroout"))
async def cmd_zeroout(message: types.Message):
    if not await is_registered(message.from_user.id):
        return await message.reply("⚠️ Please register in my DMs first to start games.")
    
    if message.chat.id in active_games:
        return await message.reply("🛑 An arena match is already active here!")

    args = message.text.split()
    if len(args) < 2 or not args[1].isdigit():
        return await message.reply("❌ <b>Invalid Usage:</b>\n▶️ <code>/zeroout 50</code> (Starts countdown from 50)")
    
    start_total = int(args[1])
    if start_total < 5:
        return await message.reply("⚠️ Starting number must be 5 or higher.")

    has_pin_rights = await can_bot_pin(message.chat.id)
    
    ui_text = f"""📉 <b>ℤ𝔼ℝ𝕆 𝕆𝕌𝕋 𝔸ℝ𝔼ℕ𝔸</b>
Host: <a href="tg://user?id={message.from_user.id}">{get_player_name(message.from_user)}</a>

🔢 <b>Current Total:</b> <code>{start_total}</code>
<i>Take turns subtracting. Do not hit 0!</i>
"""
    if not has_pin_rights:
        ui_text += "\n\n<i>⚠️ Note: I am not an Admin. Game board will not be pinned.</i>"

    sent_msg = await message.answer(ui_text, reply_markup=get_zeroout_kb())
    
    if has_pin_rights:
        try:
            await bot.pin_chat_message(message.chat.id, sent_msg.message_id)
        except Exception:
            pass

    active_games[message.chat.id] = {
        "type": "zo",
        "host": message.from_user.id,
        "msg_id": sent_msg.message_id,
        "total": start_total,
        "last_player": None,
        "last_action": time.time()
    }

# --- TEXT LISTENER (HOT & COLD) ---
@dp.message(F.text.regexp(r'^\d+$'))
async def handle_guesses(message: types.Message):
    chat_id = message.chat.id
    if chat_id not in active_games or active_games[chat_id]["type"] != "hc":
        return

    game = active_games[chat_id]
    guess = int(message.text)
    game["last_action"] = time.time()
    
    name = get_player_name(message.from_user)
    
    if guess == game["target"]:
        # Game Won
        ui_text = f"""🎯 <b>ℍ𝕆𝕋 & ℂ𝕆𝕃𝔻 𝔸ℝ𝔼ℕ𝔸</b>
Target Found: <code>{game["target"]}</code>

👑 <b>Winner:</b> <a href="tg://user?id={message.from_user.id}">{name}</a>!"""
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=game["msg_id"], text=ui_text)
            await bot.unpin_chat_message(chat_id=chat_id, message_id=game["msg_id"])
        except Exception:
            pass
        await message.reply("⚡ Bullseye!")
        del active_games[chat_id]
        return

    # Process Hint
    hint = "🔼 Higher" if guess < game["target"] else "🔽 Lower"
    history_line = f"<code>[ {guess} ]</code> {hint} — <a href=\"tg://user?id={message.from_user.id}\">{name}</a>"
    
    if game["latest_guess"]:
        game["history"].insert(0, game["latest_guess"])
        if len(game["history"]) > 5:
            game["history"].pop()
            
    game["latest_guess"] = history_line

    # Throttled Edit (Anti-Flood: Only update if 1.5s passed since last edit)
    if time.time() - game["last_edit"] >= 1.5:
        history_block = "\n".join(game["history"])
        ui_text = f"""🎯 <b>ℍ𝕆𝕋 & ℂ𝕆𝕃𝔻 𝔸ℝ𝔼ℕ𝔸</b>
Target Range: <code>1 - {game["max"]}</code>

<blockquote expandable>📜 <b>Guess History:</b>
{history_block}</blockquote>

⚡ <b>Latest:</b> {game["latest_guess"]}"""
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=game["msg_id"], text=ui_text)
            game["last_edit"] = time.time()
        except TelegramBadRequest:
            pass

# --- BUTTON LISTENER (ZERO OUT) ---
@dp.callback_query(F.data.startswith("zo_"))
async def handle_zeroout(callback: types.CallbackQuery):
    chat_id = callback.message.chat.id
    if chat_id not in active_games or active_games[chat_id]["type"] != "zo":
        return await callback.answer("⚠️ This game has ended.", show_alert=True)

    game = active_games[chat_id]
    
    # Stale Message Protection
    if callback.message.message_id != game["msg_id"]:
        return await callback.answer("⚠️ Stale board. Play on the active board.", show_alert=True)

    # Anti-Spam (Cannot play twice in a row)
    if game["last_player"] == callback.from_user.id:
        return await callback.answer("🛑 You cannot make two moves in a row!", show_alert=True)

    amount = int(callback.data.split("_")[1])
    game["total"] -= amount
    game["last_player"] = callback.from_user.id
    game["last_action"] = time.time()
    
    name = get_player_name(callback.from_user)

    if game["total"] <= 0:
        ui_text = f"""💀 <b>ℤ𝔼ℝ𝕆 𝕆𝕌𝕋 — 𝔾𝔸𝕄𝔼 𝕆𝕍𝔼ℝ</b>

🔢 <b>Final Total:</b> <code>0</code>

💥 The core collapsed! 
<tg-spoiler><a href="tg://user?id={callback.from_user.id}">{name}</a></tg-spoiler> made the final subtraction."""
        try:
            await callback.message.edit_text(text=ui_text)
            await bot.unpin_chat_message(chat_id=chat_id, message_id=game["msg_id"])
        except Exception:
            pass
        del active_games[chat_id]
        return await callback.answer("💀 You zeroed out!")

    # Standard Turn Update
    ui_text = f"""📉 <b>ℤ𝔼ℝ𝕆 𝕆𝕌𝕋 𝔸ℝ𝔼ℕ𝔸</b>
🔢 <b>Current Total:</b> <code>{game["total"]}</code>

⚡ <i>Last action: <a href="tg://user?id={callback.from_user.id}">{name}</a> subtracted {amount}</i>"""
    
    try:
        await callback.message.edit_text(text=ui_text, reply_markup=get_zeroout_kb())
    except TelegramBadRequest:
        pass
        
    await callback.answer()

# --- COMMAND MENU UPLOAD ---
async def set_bot_commands():
    commands = [
        BotCommand(command="start", description="Open the Arena Lobby"),
        BotCommand(command="nguess", description="Start Hot & Cold (e.g., /nguess 1000)"),
        BotCommand(command="zeroout", description="Start Zero Out (e.g., /zeroout 50)"),
        BotCommand(command="endgame", description="Force stop the active match")
    ]
    # Pushes the commands to Telegram
    await bot.set_my_commands(commands)

# --- MAIN ENGINE RUNNER ---
async def main():
    logging.basicConfig(level=logging.INFO)
    await init_db()
    
    print("⚡ Arena Engine Online...")
    
    # Auto-upload the command menu
    await set_bot_commands()
    
    await bot.delete_webhook(drop_pending_updates=True)
    
    # Start the background sweeper for inactive games
    asyncio.create_task(inactivity_sweeper())
    
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Arena Engine Offline.")

