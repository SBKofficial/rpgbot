import asyncio
import random
import re
import os
from telethon import TelegramClient, events
from PIL import Image, ImageOps
import pytesseract

# --- CONFIGURATION ---
API_ID = 26759620              # Replace with your my.telegram.org API ID
API_HASH = 'e5c2cfff7011b7fee949ed8293bafde8'     # Replace with your my.telegram.org API Hash
BOT_USERNAME = '@FOREST_OF_SAVIOUR_BOT' # Replace with the bot's username (e.g., '@my_game_bot')

client = TelegramClient('my_auto_bot', API_ID, API_HASH)
is_running = False

def extract_text_from_image(image_path):
    """Preprocesses the dark image for perfect OCR accuracy."""
    try:
        img = Image.open(image_path).convert('L') # Grayscale
        img = ImageOps.invert(img) # Invert (black text on white background is easier for OCR)
        
        # High contrast threshold
        img = img.point(lambda p: 255 if p > 150 else 0) 
        
        # psm 7 tells Tesseract to expect a single uniform line of text
        text = pytesseract.image_to_string(img, config='--psm 7').strip()
        return text
    finally:
        # Clean up the downloaded image so your host doesn't run out of storage
        if os.path.exists(image_path):
            os.remove(image_path)

# --- CONTROL PANEL (Listen to your own messages) ---
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
async def game_handler(event):
    if not is_running:
        return

    text = event.raw_text or ""
    
    # Global delay of 1 to 1.5 seconds per action to mimic human speed
    await asyncio.sleep(random.uniform(1.0, 1.5))

    try:
        # 1. ENCOUNTERED / CHARACTER
        if any(kw in text for kw in ["Encountered", "Character"]):
            # click(0) clicks the very first button (flat index 0)
            await event.message.click(0)

        # 2. EXPLORE TRIGGERS
        elif any(kw in text for kw in ["Victory", "Crate", "Defeat", "Correct", "Tailed"]):
            await client.send_message(BOT_USERNAME, "/explore")

        # 3. WHACK-A-MOLE
        elif "Whack-a-mole" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            # Extract just the digit from "TARGET: 7"
            match = re.search(r'\d', ocr_text) 
            if match:
                target_num = int(match.group(0))
                # Buttons are 1-9. Telethon uses 0-indexed flat arrays.
                # Button 1 = index 0. Button 7 = index 6.
                await event.message.click(target_num - 1)

        # 4. CAPTCHA
        elif "Captcha" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            # Clean up the OCR text to be strictly alphanumeric (removes random symbols)
            captcha_code = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
            
            if captcha_code and event.message.buttons:
                # Iterate through matrix to find the matching button
                for row_idx, row in enumerate(event.message.buttons):
                    for col_idx, btn in enumerate(row):
                        if captcha_code in btn.text.upper():
                            await event.message.click(row_idx, col_idx)
                            return

        # 5. POKE-SELECTION
        elif "poke-selection" in text and event.photo:
            photo_path = await event.download_media()
            ocr_text = extract_text_from_image(photo_path)
            
            # Extract the word after "PICK:" or "PICK "
            match = re.search(r'(?i)pick[:\s]*([a-z]+)', ocr_text)
            if match:
                target_poke = match.group(1).upper()
                
                if event.message.buttons:
                    for row_idx, row in enumerate(event.message.buttons):
                        for col_idx, btn in enumerate(row):
                            if target_poke in btn.text.upper():
                                await event.message.click(row_idx, col_idx)
                                return

    except Exception as e:
        print(f"Action failed on message: {text[:20]}... Error: {e}")

print("Userbot is running...")
print("Go to your 'Saved Messages' in Telegram and send '1' to start, or '0' to stop.")
client.start()
client.run_until_disconnected()

