import os
import asyncio
import requests
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# --- CONFIGURATION ---
API_ID = 1747534 # REPLACE THIS!
API_HASH = '5a2684512006853f2e48aca9652d83ea' # REPLACE THIS!
SESSION_STRING = '1BVtsOHoBu6UrVWlKWytZo4dWB7FSrJ7Va5j-Xg7kVLTE3foejEoLbVzW1kr3k145esPhWiEct-t3jqlIIt1_iBvM8qyFMP3-6k7ZeuvOOuNXVCvpeWi9c9_xsUQJtgHsvmWdJu6CpNnR5Itmr_RxCwVZqLJSYXsu7y5HpjM5XUm_BZlY5YQt53jLnj_ZvarIN4_EeU-e8dXVYYD3CHckx-CDm6xs4xyyO1oQ_P4RHjeBPZt7C58SF9YwSuYNOA_2azbg-4yxefIbJLccnnM6xSuvmlJW5Ftc2XUzSv3CUppY0lZJRNNCn5Efp6DKEEwjpqsU19xboUIDrWCwg0Dik87Y9oX9Cbs=' 

GAME_BOT_USERNAME = 'Naruto_tcg_bot'
TARGET_GROUPS = [-1003923986174, -1003531986896] # Add multiple group IDs here

bot_stats = {
    "spawns_detected": 0,
    "catch_attempts": 0,
    "ocr_failures": 0
}

client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)

@client.on(events.NewMessage(outgoing=True, pattern=r'(?i)^\.stop$'))
async def stop_command(event):
    await event.edit("🛑 **Shutting down Shinobi Catcher...**")
    print("🛑 Stop command received. Disconnecting...")
    await client.disconnect()

@client.on(events.NewMessage(outgoing=True, pattern=r'(?i)^/stats$'))
async def stats_command(event):
    stats_msg = (
        "📊 **Shinobi Catcher Stats** 📊\n\n"
        f"🍥 **Total Spawns Seen:** `{bot_stats['spawns_detected']}`\n"
        f"🎯 **Catch Attempts:** `{bot_stats['catch_attempts']}`\n"
        f"❌ **Read Failures:** `{bot_stats['ocr_failures']}`"
    )
    await event.edit(stats_msg)

# Pass the list of TARGET_GROUPS here
@client.on(events.NewMessage(chats=TARGET_GROUPS, from_users=GAME_BOT_USERNAME))
async def shinobi_catcher(event):
    if 'ᴀ sʜɪɴᴏʙɪ ʜᴀs ᴀᴘᴘᴇᴀʀᴇᴅ!' in event.raw_text:
        bot_stats["spawns_detected"] += 1
        
        # Get the specific group name or ID for logging
        chat = await event.get_chat()
        chat_title = chat.title if hasattr(chat, 'title') else event.chat_id
        
        print(f"\n🚨 Shinobi Spawn Detected in [{chat_title}]! Downloading image...")
        
        if event.message.media:
            image_path = await event.download_media(file="shinobi_spawn.jpg")
            try:
                with open(image_path, 'rb') as f:
                    payload = {
                        'isOverlayRequired': False,
                        'apikey': 'helloworld',
                        'language': 'eng'
                    }
                    response = requests.post(
                        'https://api.ocr.space/parse/image', 
                        files={image_path: f}, 
                        data=payload
                    )
                    
                result = response.json()
                
                if not result.get('IsErroredOnProcessing') and result.get('ParsedResults'):
                    extracted_text = result['ParsedResults'][0]['ParsedText']
                    
                    lines = extracted_text.split('\n')
                    valid_lines = [line.strip() for line in lines if line.strip()]
                    
                    if valid_lines:
                        shinobi_name = valid_lines[0]
                        shinobi_name = "".join(c for c in shinobi_name if c.isalpha() or c.isspace())
                        
                        catch_command = f'/catch {shinobi_name.title()}'
                        
                        # --- THE FIX: Reply directly to the specific group ---
                        sent_msg = await event.respond(catch_command)
                        # -----------------------------------------------------
                        
                        bot_stats["catch_attempts"] += 1
                        print(f"✅ Successfully Sent: {catch_command} to [{chat_title}]")
                        
                    else:
                        bot_stats["ocr_failures"] += 1
                        print("❌ OCR API returned empty text.")
                else:
                    bot_stats["ocr_failures"] += 1
                    print(f"❌ OCR API Error: {result.get('ErrorMessage', 'Unknown error')}")
                
            except Exception as e:
                bot_stats["ocr_failures"] += 1
                print(f"⚠️ Error reading image: {e}")
            finally:
                if os.path.exists(image_path):
                    os.remove(image_path)

print("Connecting to Telegram...")
client.start()

print("✅ Bot running with Public API Vision enabled!")
print(f"✅ Monitoring {len(TARGET_GROUPS)} groups.")
try:
    client.run_until_disconnected()
except Exception as e:
    print(f"Bot stopped: {e}")

