import asyncio
from telethon import TelegramClient, events

# Add your API details here
api_id = 26759620
api_hash = 'e5c2cfff7011b7fee949ed8293bafde8'

bot_id = 8691897182
XBot = TelegramClient("XBot", api_id=api_id, api_hash=api_hash)

# Global states
is_running = False
explore_count = 0

async def send_explore():
    global explore_count, is_running
    
    if not is_running:
        return
        
    if explore_count >= 500:
        is_running = False
        print("Reached limit of 500 explores. Auto-stopping.")
        return

    # Wait 1 second before sending (handles both start and caught/ran delays)
    await asyncio.sleep(1)
    await XBot.send_message(bot_id, "/explore")
    explore_count += 1
    print(f"Sent /explore. Total count: {explore_count}")

@XBot.on(events.NewMessage(outgoing=True))
async def toggle_bot(event):
    global is_running, explore_count
    
    # Send "1" anywhere to start
    if event.raw_text == "1" and not is_running:
        is_running = True
        explore_count = 0  # Resets counter on a fresh start
        print("Bot started.")
        await send_explore()
        
    # Send "0" anywhere to stop
    elif event.raw_text == "0" and is_running:
        is_running = False
        print("Bot stopped by user.")

@XBot.on(events.NewMessage(from_users=bot_id))
async def handle_new_message(event):
    if not is_running:
        return
        
    # If "appeared" is in the new message text, click 0,0 after 1s
    if "appeared" in event.raw_text.lower():
        if event.buttons:
            await asyncio.sleep(1)
            try:
                await event.click(0, 0)
                print("Clicked button (0, 0)")
            except Exception as e:
                print(f"Error clicking button: {e}")

@XBot.on(events.MessageEdited(from_users=bot_id))
async def handle_message_edited(event):
    if not is_running:
        return
        
    text_lower = event.raw_text.lower()
    
    # If edited text says caught or ran, wait 1s and explore again
    if "caught" in text_lower or "ran" in text_lower:
        print("Pokemon caught or ran. Continuing...")
        await send_explore()

async def main():
    await XBot.start()
    print("Script is running! Send '1' in any chat to start the loop, and '0' to pause/stop.")
    await XBot.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())

