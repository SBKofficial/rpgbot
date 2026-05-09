import io
from PIL import Image, ImageOps
import asyncio
import unicodedata
import re
import os
import random
import difflib
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
    """100% Offline OCR optimized for White Backgrounds."""
    try:
        print("Reading image locally with ddddocr...")
        start_time = time.time()
        
        # 1. PREPROCESS THE IMAGE (White Background Mode)
        img = Image.open(photo_path)
        
        # Make it 2x larger
        new_size = (img.width * 2, img.height * 2)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to Grayscale
        img = img.convert('L')
        
        # MAGIC THRESHOLD: The background is white (~255), text is colored (darker).
        # This forces all colored text to pure Black (0) and background to pure White (255)
        img = img.point(lambda p: 0 if p < 230 else 255) 
        
        # 2. AUTO-CROP THE EMPTY WHITE SPACE
        # Pillow's getbbox() needs a black background to find the text, so we invert temporarily
        inverted_for_bbox = ImageOps.invert(img) 
        bbox = inverted_for_bbox.getbbox()
        
        if bbox:
            # Crop the image to exactly where the text is
            img = img.crop(bbox)
            # Add a small 10-pixel white border so the letters don't touch the edge
            img = ImageOps.expand(img, border=10, fill='white')
            
        # Save debug copy (Should look like incredibly crisp black text on a tiny white box)
        img.save("debug_image.jpg")
        
        # 3. Convert to bytes for ddddocr
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
            
        def _read():
            return ocr_engine.classification(image_bytes)
            
        # 4. Read the image
        result = await asyncio.wait_for(asyncio.to_thread(_read), timeout=45.0)
        
        elapsed = time.time() - start_time
        clean_result = result.lower().strip()
        
        print(f"✅ OCR finished in {elapsed:.2f} seconds!")
        print(f"🔍 OCR Saw: '{clean_result}'") 
        
        return clean_result
        
    except asyncio.TimeoutError:
        print("⚠️ CRITICAL: OCR took more than 45 seconds!")
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
    global is_running  # <--- Added so we can turn the bot off from here
    
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
        if any(kw in clean_text for kw in ["slot", "hit", "pokemon", "code"]) and event.photo:
            print("▶ Group 1: Image Challenge Detected")
            photo_path = await event.download_media()
            ocr_text = await extract_text_from_image(photo_path)
            
            clicked = False
            
            # 1. SLOT / HIT (Whack-a-mole)
            if "slot" in clean_text or "hit" in clean_text:
                match = re.search(r'\d', ocr_text) 
                if match:
                    target_num = int(match.group(0))
                    
                    if 1 <= target_num <= 9:
                        row_idx = (target_num - 1) // 3
                        col_idx = (target_num - 1) % 3
                        
                        delay = random.uniform(1.0, 2.0)
                        print(f"Sleeping {delay:.2f}s before clicking HIT ({row_idx}, {col_idx})...")
                        await asyncio.sleep(delay)
                        
                        await event.message.click(row_idx, col_idx)
                        clicked = True
                    else:
                        print(f"⚠️ OCR read an invalid number: {target_num}")
                        
            # 2. POKEMON
            elif "pokemon" in clean_text:
                target_poke = re.sub(r'(?i)(pick|pck|pic|piks|pcks|pcks:)[^a-z]*', '', ocr_text).strip()
                if not target_poke:
                    target_poke = ocr_text

                if event.message.buttons:
                    button_texts = [btn.text.lower() for row in event.message.buttons for btn in row]
                    matches = difflib.get_close_matches(target_poke, button_texts, n=1, cutoff=0.3)
                    
                    if matches:
                        best_match = matches[0]
                        print(f"🎯 POKEMON MATCH SUCCESS! OCR read '{ocr_text}' -> Matched to '{best_match}'")
                        
                        for row_idx, row in enumerate(event.message.buttons):
                            for col_idx, btn in enumerate(row):
                                if btn.text.lower() == best_match:
                                    delay = random.uniform(1.0, 2.0)
                                    print(f"Sleeping {delay:.2f}s before clicking Pokemon...")
                                    await asyncio.sleep(delay)
                                    await event.message.click(row_idx, col_idx)
                                    clicked = True
                                    break
                            if clicked: break
                    else:
                        print(f"⚠️ Pokemon Match Failed. Could not link '{target_poke}' to any buttons.")

            # 3. CODE (Captcha)
            elif "code" in clean_text:
                captcha_code = re.sub(r'[^a-z0-9]', '', ocr_text)
                
                if captcha_code and event.message.buttons:
                    button_texts = [btn.text.lower() for row in event.message.buttons for btn in row]
                    matches = difflib.get_close_matches(captcha_code, button_texts, n=1, cutoff=0.5)
                    
                    if matches:
                        best_match = matches[0]
                        print(f"🎯 CAPTCHA MATCH SUCCESS! OCR read '{captcha_code}' -> Matched to '{best_match}'")
                        
                        for row_idx, row in enumerate(event.message.buttons):
                            for col_idx, btn in enumerate(row):
                                if btn.text.lower() == best_match:
                                    delay = random.uniform(1.0, 2.0)
                                    print(f"Sleeping {delay:.2f}s before clicking Captcha...")
                                    await asyncio.sleep(delay)
                                    await event.message.click(row_idx, col_idx)
                                    clicked = True
                                    break
                            if clicked: break
                    else:
                        print(f"⚠️ Captcha Match Failed. Could not link '{captcha_code}' to any buttons.")

            # --- THE NEW FAILSAFE ---
            if not clicked:
                print("⚠️ OCR failed to match anything. STOPPING SCRIPT.")
                is_running = False  # Turn off the main loop
                # Send a message to yourself so you get a ping on your phone
                await client.send_message('me', "🔴 **FAILSAFE TRIGGERED:** OCR failed to read the image. Auto-script has been STOPPED.")
                
            return 

        # ==========================================
        # GROUP 2: COMBAT ENCOUNTERS
        # ==========================================
        elif any(kw in clean_text for kw in ["yourself","wander", "trembles"]):
            print("▶ Group 2: Combat Detected.")
            delay = random.uniform(1.0, 2.0)
            print(f"Sleeping {delay:.2f}s before clicking Combat (0,0)...")
            await asyncio.sleep(delay)
            await event.message.click(0, 0)

        # ==========================================
        # GROUP 3: LOOT, DEFEAT & SKIPS
        # ==========================================
        elif any(kw in clean_text for kw in ["solved", "crate", "seal", "reward", "were", "smiles", "retreated"]):
            current_time = time.time()
            
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

