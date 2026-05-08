import asyncio
import random
import re
import os
from telethon import TelegramClient, events
from PIL import Image, ImageOps
import pytesseract

# --- CONFIGURATION ---
API_ID = 26759620
API_HASH = 'e5c2cfff7011b7fee949ed8293bafde8'
BOT_USERNAME = '@FOREST_OF_SAVIOUR_BOT' 

client = TelegramClient('my_auto_bot', API_ID, API_HASH)
is_running = False

def extract_text_from_image(image_path):
    """Preprocesses the dark image and returns lowercase text for safe matching."""
    try:
        img = Image.open(image_path).convert('L')
        img = ImageOps.invert(img) 
        img = img.point(lambda p: 255 if p > 150 else 0) 
        
        text = pytesseract.image_to_string(img, config='--psm 7').strip()
        return text.lower() # Return everything in lowercase
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# --- CONTROL PANEL ---
@client.on(events.NewMessage(chats='me'))
async def toggle_script(event):
    global is_running
    if event.raw_text == '1':
        is_running = True
        await event.reply("🟢 Auto-script STARTED.")
    elif event.raw_text == '0':
        is_running = False
        await event.reply("🔴 Auto-script STOPPED.")

# --- MAIN GAME LOGIC ---
@client.on(events.NewMessage(chats=BOT_USERNAME))
@client.on(events.MessageEdited(chats=BOT_USERNAME))
async def game_handler(event):
    if not is_running:
        return

    # 1. Convert the entire incoming message to lowercase immediately
    text = (event.raw_text or "").lower()
    
    await asyncio.sleep(random.uniform(1.0, 1.5))

    try:
        # 2. ENCOUNTERED / CHARACTER
        if any(kw in text for kw in ["encountered", "character"]):
            await event.message.click(0)

        # 3. EXPLORE TRIGGERS
        elif any(kw in text for kw in ["victory", "crate", "defeat", "correct", "tailed"]):
            await client.send_message(BOT_USERNAME, "/explore")

        # 4. WHACK-A-MOLE
        elif "whack-a-mole" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            match = re.search(r'\d', ocr_text) 
            if match:
                target_num = int(match.group(0))
                await event.message.click(target_num - 1)

        # 5. CAPTCHA
        elif "captcha" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            # Clean up to strictly lowercase alphanumeric
            captcha_code = re.sub(r'[^a-z0-9]', '', ocr_text)
            
            if captcha_code and event.message.buttons:
                for row_idx, row in enumerate(event.message.buttons):
                    for col_idx, btn in enumerate(row):
                        # Convert button text to lowercase before checking
                        if captcha_code in btn.text.lower():
                            await event.message.click(row_idx, col_idx)
                            return

        # 6. POKE-SELECTION
        elif "poke-selection" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            # Extracts words, spaces, or hyphens after "pick:"
            match = re.search(r'pick[:\s]*([a-z\s\-]+)', ocr_text)
            if match:
                target_poke = match.group(1).strip()
                
                if event.message.buttons:
                    for row_idx, row in enumerate(event.message.buttons):
                        for col_idx, btn in enumerate(row):
                            # Convert button text to lowercase before checking
                            if target_poke in btn.text.lower():
                                await event.message.click(row_idx, col_idx)
                                return

    except Exception as e:
        print(f"Action failed on message: {text[:20]}... Error: {e}")

print("Userbot is running...")
print("Go to your 'Saved Messages' in Telegram and send '1' to start, or '0' to stop.")
client.start()
client.run_until_disconnected()

