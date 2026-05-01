import asyncio
import re
import random
import os
import cv2
import numpy as np
from telethon import TelegramClient, events

# Add your API details here
api_id = 26759620
api_hash = 'e5c2cfff7011b7fee949ed8293bafde8'
bot_id = 8709023864  # Updated Target Bot ID

XBot = TelegramClient("XBot", api_id=api_id, api_hash=api_hash)

# --- CONFIGURATION ---
stop_pokemon = ["Zacian", "Zamazenta", "Eternatus"] 
stop_words = ["✨️"] 

is_hunting = False

def solve_captcha(image_path):
    """
    Reads the downloaded captcha image, extracts the top pokemon,
    matches it to the grid below, and returns the correct button number (1-6).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    # 1. Crop a tight box right in the middle of the top image.
    # This avoids the blue border and grabs the core pixels of the Pokemon.
    template = img[int(h * 0.12):int(h * 0.28), int(w * 0.42):int(w * 0.58)]
    
    # 2. Isolate the bottom half of the image where the 6 options are.
    bottom_half = img[int(h * 0.35):h, :]
    
    # 3. Match the template to the bottom half
    res = cv2.matchTemplate(bottom_half, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    best_x, best_y = max_loc
    
    # 4. Find the center of our match to determine which quadrant it is in
    match_center_x = best_x + (template.shape[1] // 2)
    match_center_y = best_y + (template.shape[0] // 2)
    
    # Calculate column (0, 1, or 2) and row (0 or 1)
    col_width = w // 3
    row_height = bottom_half.shape[0] // 2
    
    col = match_center_x // col_width
    row = match_center_y // row_height
    
    # Calculate button number (1 through 6)
    ans = (row * 3) + col + 1
    
    # Safety clamp just in case math goes slightly out of bounds
    return max(1, min(6, ans))


# This handler ONLY listens to your "Saved Messages"
@XBot.on(events.NewMessage(chats='me'))
async def toggle_bot(event):
    global is_hunting
    text = event.raw_text.strip()
    
    # Send "1" in Saved Messages to start
    if text == "1" and not is_hunting:
        is_hunting = True
        print("Bot started! Sending first /hunt...")
        await XBot.send_message(bot_id, "/hunt")
        
    # Send "0" in Saved Messages to stop
    elif text == "0" and is_hunting:
        is_hunting = False
        print("Bot stopped by user.")


@XBot.on(events.NewMessage(from_users=bot_id))
async def handle_new_message(event):
    global is_hunting
    if not is_hunting:
        return

    text = event.raw_text
    text_lower = text.lower()

    # --- CAPTCHA TRIGGER ---
    if "officer jenny: hello" in text_lower and "stopping for a bit" in text_lower:
        print("🚔 Officer Jenny encountered! Attempting to solve Captcha...")
        if event.photo:
            # Download the image
            photo_path = await event.download_media('captcha.jpg')
            
            # Solve it
            ans = solve_captcha(photo_path)
            print(f"Captcha solved! The matching Pokemon is in position: {ans}")
            
            # Click the corresponding button dynamically
            if event.buttons:
                for row_idx, row_buttons in enumerate(event.buttons):
                    for col_idx, button in enumerate(row_buttons):
                        if button.text == str(ans):
                            await asyncio.sleep(1) # Small delay to mimic human behavior
                            await event.click(row_idx, col_idx)
                            print(f"Clicked button [{ans}]")
                            
            # Clean up the downloaded image
            if os.path.exists(photo_path):
                os.remove(photo_path)
        return

    # --- CAPTCHA SUCCESS TRIGGER ---
    if "apologize for the interruption" in text_lower:
        print("✅ Captcha passed successfully! Resuming hunt...")
        delay = random.uniform(1.0, 1.5)
        await asyncio.sleep(delay)
        await XBot.send_message(bot_id, "/hunt")
        return

    # 1. Check for Stop Words anywhere in the text
    if any(word.lower() in text_lower for word in stop_words):
        print("Stop word found in message! Stopping the hunt.")
        is_hunting = False
        return

    # 2. Check for Target Pokemon anywhere in the text
    if any(p.lower() in text_lower for p in stop_pokemon):
        print("Target Legendary found! Stopping the hunt.")
        is_hunting = False
        return

    # 3. Extract and print the encountered Pokemon name for logging
    match = re.search(r"(?i)a wild\s+(.*?)\s+\(Lv\.\s*\d+\)\s+appeared!", text)
    if match:
        pokemon_name = match.group(1).strip()
        clean_name = pokemon_name.replace("✨️", "").strip()
        print(f"Encountered: {clean_name}")

    # 4. If no stop conditions were met, wait 1-1.5s and continue hunting
    delay = random.uniform(1.0, 1.5)
    await asyncio.sleep(delay)
    await XBot.send_message(bot_id, "/hunt")


async def main():
    await XBot.start()
    print("Script started!")
    print(" - Send '1' in your Saved Messages to start the hunt.")
    print(" - Send '0' in your Saved Messages to stop the hunt.")
    
    await XBot.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())

