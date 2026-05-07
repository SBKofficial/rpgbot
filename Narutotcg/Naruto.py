import os
import asyncio
import requests
import hashlib
import unicodedata
import json
import re
from telethon import TelegramClient, events
from telethon.sessions import StringSession

# --- CONFIGURATION ---
API_ID = 1747534 
API_HASH = '5a2684512006853f2e48aca9652d83ea' 
SESSION_STRING = '1BVtsOHoBu6UrVWlKWytZo4dWB7FSrJ7Va5j-Xg7kVLTE3foejEoLbVzW1kr3k145esPhWiEct-t3jqlIIt1_iBvM8qyFMP3-6k7ZeuvOOuNXVCvpeWi9c9_xsUQJtgHsvmWdJu6CpNnR5Itmr_RxCwVZqLJSYXsu7y5HpjM5XUm_BZlY5YQt53jLnj_ZvarIN4_EeU-e8dXVYYD3CHckx-CDm6xs4xyyO1oQ_P4RHjeBPZt7C58SF9YwSuYNOA_2azbg-4yxefIbJLccnnM6xSuvmlJW5Ftc2XUzSv3CUppY0lZJRNNCn5Efp6DKEEwjpqsU19xboUIDrWCwg0Dik87Y9oX9Cbs=' 

GAME_BOT_USERNAME = 'Naruto_tcg_bot'
TARGET_GROUPS = [-1003923986174, -1003531986896] 

# --- 🎰 BETTING CONFIG 🎰 ---
BETTING_GROUP_ID = -1003531986896  # REPLACE THIS with the actual betting group ID!
is_betting = False

bot_stats = {
    "spawns_detected": 0,
    "catch_attempts": 0,
    "ocr_failures": 0,
    "cache_hits": 0
}

# --- 🧠 MEMORY CACHE SETUP 🧠 ---
CACHE_FILE = 'shinobi_cache.json'

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        image_cache = json.load(f)
    print(f"🧠 Loaded {len(image_cache)} known shinobi into memory.")
else:
    image_cache = {}

def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(image_cache, f)

client = TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH)

# --- 🛠️ BASE COMMANDS 🛠️ ---

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
        f"⚡ **Cache Hits (Instant Catches):** `{bot_stats['cache_hits']}`\n"
        f"❌ **Read Failures:** `{bot_stats['ocr_failures']}`"
    )
    await event.edit(stats_msg)

@client.on(events.NewMessage(outgoing=True, pattern=r'(?i)^\.fixname (.*) \| (.*)'))
async def fix_name_command(event):
    old_name = event.pattern_match.group(1).strip()
    new_name = event.pattern_match.group(2).strip()
    
    updated_count = 0
    for img_hash, cached_name in image_cache.items():
        if cached_name.lower() == old_name.lower():
            image_cache[img_hash] = new_name
            updated_count += 1
            
    if updated_count > 0:
        save_cache()
        await event.edit(f"✅ **Cache Updated!**\nChanged `{old_name}` to `{new_name}` across {updated_count} image(s).")
        print(f"🛠️ Cache manually corrected: {old_name} -> {new_name}")
    else:
        await event.edit(f"⚠️ **Not Found:** Could not find `{old_name}` in the local cache.")

# --- 🎰 BETTING COMMANDS & LOOP 🎰 ---

@client.on(events.NewMessage(outgoing=True, pattern=r'(?i)^\.betting$'))
async def start_betting_command(event):
    global is_betting
    is_betting = True
    await event.edit("🎰 **Betting Loop Started!** Fetching initial balance...")
    print("🎰 Betting started! Sending /shinobi_coins...")
    
    await client.send_message(BETTING_GROUP_ID, '/shinobi_coins')

@client.on(events.NewMessage(outgoing=True, pattern=r'(?i)^\.enough$'))
async def stop_betting_command(event):
    global is_betting
    is_betting = False
    await event.edit("🛑 **Betting Loop Stopped.**")
    print("🛑 Betting loop manually halted.")

@client.on(events.NewMessage(chats=BETTING_GROUP_ID, from_users=GAME_BOT_USERNAME))
async def auto_better(event):
    global is_betting
    
    if not is_betting:
        return

    text = event.raw_text
    
    match = re.search(r'(?:Balance|Remaining):\s*([\d,]+)\s*coins', text, re.IGNORECASE)
    
    if match:
        current_balance = int(match.group(1).replace(',', ''))
        bet_amount = current_balance // 2
        
        if bet_amount > 0:
            print(f"💰 Balance detected: {current_balance}. Betting {bet_amount}...")
            await asyncio.sleep(2) 
            
            bet_command = f'/bet {bet_amount} h'
            await client.send_message(BETTING_GROUP_ID, bet_command)
            print(f"🎰 Sent: {bet_command}")
            
        else:
            is_betting = False
            print("🛑 Balance is too low to continue halving. Auto-stopping the betting loop.")
            await client.send_message(BETTING_GROUP_ID, "🛑 Balance too low. Betting loop stopped.")

# --- 🍥 SHINOBI CATCHER 🍥 ---

@client.on(events.NewMessage(chats=TARGET_GROUPS, from_users=GAME_BOT_USERNAME))
async def shinobi_catcher(event):
    if 'ᴀ sʜɪɴᴏʙɪ ʜᴀs ᴀᴘᴘᴇᴀʀᴇᴅ!' in event.raw_text:
        bot_stats["spawns_detected"] += 1
        
        chat = await event.get_chat()
        chat_title = chat.title if hasattr(chat, 'title') else event.chat_id
        
        print(f"\n🚨 Shinobi Spawn Detected in [{chat_title}]! Downloading image...")
        
        if event.message.media:
            image_path = await event.download_media(file="shinobi_spawn.jpg")
            try:
                # 1. Generate MD5 hash
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                img_hash = hashlib.md5(image_bytes).hexdigest()
                
                # 2. Check memory cache first
                if img_hash in image_cache:
                    full_shinobi_name = image_cache[img_hash]
                    first_name = full_shinobi_name.split()[0] if full_shinobi_name else ""
                    
                    bot_stats["cache_hits"] += 1
                    print(f"⚡ Image recognized! Full Name: '{full_shinobi_name}'. Skipping OCR.")
                    
                    if first_name:
                        catch_command = f'/catch {first_name}'
                        sent_msg = await event.respond(catch_command)
                        bot_stats["catch_attempts"] += 1
                        print(f"✅ Successfully Sent: {catch_command} to [{chat_title}]")
                    
                else:
                    # 3. Use OCR API
                    print("🔍 New image! Sending to OCR API...")
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
                            raw_name = valid_lines[0]
                            
                            clean_name = unicodedata.normalize('NFKD', raw_name).encode('ASCII', 'ignore').decode('utf-8')
                            full_name = "".join(c for c in clean_name if c.isalpha() or c.isspace()).title().strip()
                            first_name = full_name.split()[0] if full_name else ""
                            
                            if first_name:
                                catch_command = f'/catch {first_name}'
                                sent_msg = await event.respond(catch_command)
                                bot_stats["catch_attempts"] += 1
                                print(f"✅ Successfully Sent: {catch_command} to [{chat_title}]")
                                
                                image_cache[img_hash] = full_name
                                save_cache()
                                print(f"💾 Saved full name '{full_name}' to local cache.")
                            else:
                                print("❌ Name extraction resulted in empty string.")
                            
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

print("✅ Bot running! Catcher, Cache, and Auto-Betting enabled.")
print(f"✅ Monitoring {len(TARGET_GROUPS)} groups for spawns.")
try:
    client.run_until_disconnected()
except Exception as e:
    print(f"Bot stopped: {e}")

