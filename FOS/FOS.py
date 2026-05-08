import io
from PIL import Image, ImageOps
import asyncio
import unicodedata
import re
import os
import random
import time
from telethon import TelegramClient, events
import ddddocr

# --- CONFIGURATION ---
API_ID = 26759620
API_HASH = 'e5c2cfff7011b7fee949ed8293bafde8'
TARGET_BOT_ID = 8015674697

# Initialize the offline OCR engine once
ocr_engine = ddddocr.DdddOcr(show_ad=False)

client = TelegramClient('my_auto_bot', API_ID, API_HASH)
is_running = False

# Watchdog & Memory Variables
response_received_event = asyncio.Event()
last_explore_time = 0 

async def extract_text_from_image(photo_path):
    """100% Offline OCR with Image Preprocessing for perfect accuracy."""
    try:
        print("Reading image locally with ddddocr...")
        start_time = time.time()
        
        # 1. PREPROCESS THE IMAGE: Invert colors and apply high-contrast
        img = Image.open(photo_path).convert('L') # Grayscale
        img = ImageOps.invert(img) # Invert darks and lights
        img = img.point(lambda p: 255 if p > 150 else 0) # High contrast
        
        # 2. Convert the processed image back to raw bytes for ddddocr
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
            
        def _read():
            return ocr_engine.classification(image_bytes)
            
        # 3. Read the image with a timeout
        result = await asyncio.wait_for(asyncio.to_thread(_read), timeout=45.0)
        
        elapsed = time.time() - start_time
        clean_result = result.lower().strip()
        
        print(f"✅ OCR finished in {elapsed:.2f} seconds!")
        print(f"🔍 OCR Saw: '{clean_result}'") # <--- DEBUG TOOL
        
        return clean_result
        
    except asyncio.TimeoutError:
        print("⚠️ CRITICAL: OCR took more than 45 seconds! Host CPU is too slow.")
        return ""
    except Exception as e:
        print(f"Local OCR Error: {e}")
        return ""
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)

# --- WATCHDOG LOGIC ---
async def send_explore():
    global is_running
    if not is_running:
        return

    delay = random.uniform(1.0, 2.0)
    print(f"Sleeping for {delay:.2f}s before exploring...")
    await asyncio.sleep(delay)
    
    print("Sending /explore...")
    await client.send_message(TARGET_BOT_ID, "/explore")
    asyncio.create_task(explore_watchdog())

async def explore_watchdog():
    response_received_event.clear()
    try:
        # Increased to 20 seconds to account for game bot lag
        await asyncio.wait_for(response_received_event.wait(), timeout=20.0)
    except asyncio.TimeoutError:
        if is_running:
            print("⚠️ Watchdog: No response for 20 seconds. Resending /explore...")
            await send_explore()

# --- CONTROL PANEL ---
@client.on(events.NewMessage(chats='me'))
async def toggle_script(event):
    global is_running
    if event.raw_text == '1':
        if not is_running:
            is_running = True
            await event.reply("🟢 Auto-script STARTED. Initiating first /explore...")
            await send_explore()
    elif event.raw_text == '0':
        is_running = False
        await event.reply("🔴 Auto-script STOPPED.")

# --- MAIN GAME HANDLER ---
@client.on(events.NewMessage(chats=TARGET_BOT_ID))
@client.on(events.MessageEdited(chats=TARGET_BOT_ID))
async def game_handler(event):
    global last_explore_time
    
    if not is_running:
        return

    # Signal the watchdog that a message arrived safely
    response_received_event.set()

    raw_text = event.raw_text or ""
    clean_text = unicodedata.normalize('NFKC', raw_text).lower().replace(" ", "")

    try:
        # ==========================================
        # GROUP 1: IMAGE CHALLENGES
        # ==========================================
        if any(kw in clean_text for kw in ["slot", "pokemon", "code"]) and event.photo:
            print("▶ Group 1: Image Challenge Detected")
            photo_path = await event.download_media()
            ocr_text = await extract_text_from_image(photo_path)
            
            clicked = False
            
            # 1. SLOT (Whack-a-mole: "Target: X")
            if "slot" in clean_text:
                match = re.search(r'\d', ocr_text) 
                if match:
                    target_num = int(match.group(0))
                    
                    # Ensure it's a valid 1-9 button
                    if 1 <= target_num <= 9:
                        # MATRIX MATH: Convert 1-9 into (Row, Column) for a 3x3 grid
                        row_idx = (target_num - 1) // 3
                        col_idx = (target_num - 1) % 3
                        
                        delay = random.uniform(1.0, 2.0)
                        print(f"Sleeping {delay:.2f}s before clicking Slot ({row_idx}, {col_idx})...")
                        await asyncio.sleep(delay)
                        
                        # Click using exact Row and Column coordinates
                        await event.message.click(row_idx, col_idx)
                        clicked = True
                    else:
                        print(f"⚠️ OCR read an invalid number: {target_num}")
                    
            elif "pokemon" in clean_text:
                match = re.search(r'pick[^a-z]*([a-z\-]+)', ocr_text)
                if match:
                    target_poke = match.group(1).strip()
                    if event.message.buttons:
                        for row_idx, row in enumerate(event.message.buttons):
                            for col_idx, btn in enumerate(row):
                                if target_poke in btn.text.lower():
                                    delay = random.uniform(1.0, 2.0)
                                    print(f"Sleeping {delay:.2f}s before clicking Pokemon...")
                                    await asyncio.sleep(delay)
                                    await event.message.click(row_idx, col_idx)
                                    clicked = True
                                    break
                            if clicked: break

            elif "code" in clean_text:
                captcha_code = re.sub(r'[^a-z0-9]', '', ocr_text)
                if captcha_code and event.message.buttons:
                    for row_idx, row in enumerate(event.message.buttons):
                        for col_idx, btn in enumerate(row):
                            if captcha_code in btn.text.lower():
                                delay = random.uniform(1.0, 2.0)
                                print(f"Sleeping {delay:.2f}s before clicking Captcha...")
                                await asyncio.sleep(delay)
                                await event.message.click(row_idx, col_idx)
                                clicked = True
                                break
                            if clicked: break
                            
            # Failsafe: if the host took so long that it failed to click
            if not clicked:
                print("⚠️ OCR failed or returned nothing. Refreshing puzzle...")
                await send_explore()

            return # Exit handler so we don't accidentally trigger Group 2 or 3

        # ==========================================
        # GROUP 2: COMBAT ENCOUNTERS
        # ==========================================
        elif any(kw in clean_text for kw in ["yourself", "trembles"]):
            print("▶ Group 2: Combat Detected.")
            delay = random.uniform(1.0, 2.0)
            print(f"Sleeping {delay:.2f}s before clicking Combat (0,0)...")
            await asyncio.sleep(delay)
            await event.message.click(0, 0)

        # ==========================================
        # GROUP 3: LOOT, DEFEAT & SKIPS
        # ==========================================
        elif any(kw in clean_text for kw in ["solved", "energy", "seal", "reward", "were", "retreated"]):
            current_time = time.time()
            
            # The 5-Second Cooldown Lock
            if current_time - last_explore_time < 5.0:
                print("▶ Group 3: Duplicate message caught by Cooldown Lock. Ignoring.")
                return 
                
            last_explore_time = current_time 
            print("▶ Group 3: Trigger hit. Sending to Explore Loop...")
            await send_explore()

    except Exception as e:
        print(f"Error handling message: {e}")

print("Userbot is running on Host (100% Offline OCR with Timers)...")
print("Go to your 'Saved Messages' in Telegram and send '1' to start, or '0' to stop.")

client.start()
client.run_until_disconnected()

