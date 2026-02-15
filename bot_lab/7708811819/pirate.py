from telethon import TelegramClient, events
import asyncio

# --- Configuration ---
api_id = 29644708
api_hash = '0db39046c635489ccb05d9a0ca395c9f'
BOT_USERNAME = 'Pirateshowdown_bot'

# --- Global State ---
automation_active = False 

client = TelegramClient("slug_session", api_id, api_hash)

# --- Control Handler (Send '0' or '1' to Saved Messages) ---
@client.on(events.NewMessage(outgoing=True, pattern=r'^[01]$'))
async def control_handler(event):
    global automation_active
    text = event.raw_text
    
    if text == '1':
        automation_active = True
        await event.respond("✅ **Pirate Showdown Automation: ON**")
        # Optional: Send initial command to start the loop
        await client.send_message(BOT_USERNAME, "/explore")

    elif text == '0':
        automation_active = False
        await event.respond("🛑 **Pirate Showdown Automation: OFF**")

# --- Game Event Handler (New Messages) ---
@client.on(events.NewMessage(chats=BOT_USERNAME))
async def on_new_message(event):
    if not automation_active:
        return

    text = event.raw_text
    
    # 1. Click first button if EXPLORATION is found
    if "EXPLORATION" in text:
        await asyncio.sleep(2)
        try:
            await event.click(0, 0)
        except Exception as e:
            print(f"Error clicking EXPLORATION: {e}")

    # 2. Send /explore if Found or stumbled is in text
    elif "found" in text or "stumbled" in text:
        await asyncio.sleep(2)
        await client.send_message(BOT_USERNAME, "/explore")


# --- Game Event Handler (Edited Messages) ---
@client.on(events.MessageEdited(chats=BOT_USERNAME))
async def on_edited_message(event):
    if not automation_active:
        return

    text = event.raw_text
    me = await client.get_me()
    first_name = me.first_name

    # Click first button if it is your turn
    if f"TURN: {first_name}" in text:
        await asyncio.sleep(2)
        try:
            await event.click(0, 0)
        except Exception as e:
            print(f"Error clicking Turn: {e}")

    elif "LOOT DROPPED" in text:
        await asyncio.sleep(2)
        await client.send_message(BOT_USERNAME, "/explore")

async def main():
    print("🚀 Pirate Bot Client Starting...")
    await client.start()
    print("✅ Connected. Send '1' to Saved Messages to begin.")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())