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

client = TelegramClient('my_auto_boot', API_ID, API_HASH)

# State Variables
is_running = False
combat_mode = 'all'

# Global Rarity Toggles
ignore_nl = False
ignore_leg = False
ignore_et = False

# Sub-Mode Slay Exceptions
sub_slay_nl = False
sub_slay_leg = False
sub_slay_et = False

# Session Stats
stats = {
    "nl": 0,
    "leg": 0,
    "et": 0,
    "unknown_char": 0,
    "monsters_sealed": 0,
    "slayed_names": []
}

# Watchdog & Memory Variables
response_received_event = asyncio.Event()
watchdog_task = None 
last_g3_time = 0 
last_g4_time = 0 

async def extract_text_from_image(photo_path):
    """100% Offline OCR optimized for White Backgrounds."""
    try:
        start_time = time.time()
        img = Image.open(photo_path)
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
        img = img.convert('L')
        img = img.point(lambda p: 0 if p < 230 else 255) 
        
        inverted_for_bbox = ImageOps.invert(img) 
        bbox = inverted_for_bbox.getbbox()
        if bbox:
            img = img.crop(bbox)
            img = ImageOps.expand(img, border=10, fill='white')
            
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
            
        def _read(): return ocr_engine.classification(image_bytes)
        result = await asyncio.wait_for(asyncio.to_thread(_read), timeout=45.0)
        
        clean_result = result.lower().strip()
        print(f"✅ OCR finished in {time.time() - start_time:.2f}s! Saw: '{clean_result}'") 
        return clean_result
    except Exception as e:
        print(f"Local OCR Error: {e}")
        return ""
    finally:
        if os.path.exists(photo_path): os.remove(photo_path)

# --- UNBREAKABLE WATCHDOG LOGIC ---
async def watchdog_worker():
    print("👁️ Watchdog active. Monitoring for bot lag...")
    while is_running:
        try:
            await asyncio.wait_for(response_received_event.wait(), timeout=20.0)
            response_received_event.clear()
        except asyncio.TimeoutError:
            if is_running:
                print("⚠️ Watchdog: No response for 20 seconds. Poking bot again...")
                try:
                    await client.send_message(TARGET_BOT_ID, "/explore")
                except Exception as e:
                    print(f"Watchdog Error: {e}")
                response_received_event.clear()
        except asyncio.CancelledError:
            break

async def trigger_explore():
    if not is_running: return
    asyncio.create_task(_delayed_explore())

async def _delayed_explore():
    if not is_running: return
    delay = random.uniform(1.0, 2.0)
    print(f"Sleeping for {delay:.2f}s before exploring...")
    await asyncio.sleep(delay)
    
    response_received_event.clear()
    print("Sending /explore...")
    try:
        await client.send_message(TARGET_BOT_ID, "/explore")
    except Exception as e:
        print(f"Error sending /explore: {e}")

# --- CONTROL PANEL ---
@client.on(events.NewMessage(chats='me'))
async def toggle_script(event):
    global is_running, combat_mode, watchdog_task
    global ignore_nl, ignore_leg, ignore_et
    global sub_slay_nl, sub_slay_leg, sub_slay_et, stats
    
    text = event.raw_text.strip().lower()
    
    if text == '1':
        if not is_running:
            is_running = True
            watchdog_task = asyncio.create_task(watchdog_worker())
            await event.reply(f"🟢 Auto-script STARTED.\n⚔️ Current Mode: **{combat_mode.upper()}**\nInitiating first /explore...")
            await trigger_explore()
        else:
            await event.reply("⚠️ Script is already running!")
            
    elif text == '0':
        if is_running:
            is_running = False
            if watchdog_task: watchdog_task.cancel()
            await event.reply("🔴 Auto-script STOPPED.")
        else:
            await event.reply("⚠️ Script is already stopped!")
        
    elif text == 'slay':
        combat_mode = 'slay'
        await event.reply("🗡️ **Mode Changed: SLAY**\nWill ONLY fight Anime Characters.")
    elif text == 'sub':
        combat_mode = 'subjugate'
        await event.reply("🧿 **Mode Changed: SUBJUGATE**\nWill ONLY seal Monsters.")
    elif text == 'all':
        combat_mode = 'all'
        await event.reply("🔄 **Mode Changed: ALL**\nWill engage everything.")

    elif text == 'ignore nl':
        ignore_nl = not ignore_nl
        await event.reply(f"🚫 Ignore Non-Legendary: **{'ON' if ignore_nl else 'OFF'}**")
    elif text == 'ignore leg':
        ignore_leg = not ignore_leg
        await event.reply(f"🚫 Ignore Legendary: **{'ON' if ignore_leg else 'OFF'}**")
    elif text == 'ignore et':
        ignore_et = not ignore_et
        await event.reply(f"🚫 Ignore Eternal: **{'ON' if ignore_et else 'OFF'}**")

    elif text == 'sub slay nl':
        sub_slay_nl = not sub_slay_nl
        await event.reply(f"⚔️ Sub-Mode Slay Non-Legendary: **{'ON' if sub_slay_nl else 'OFF'}**")
    elif text == 'sub slay leg':
        sub_slay_leg = not sub_slay_leg
        await event.reply(f"⚔️ Sub-Mode Slay Legendary: **{'ON' if sub_slay_leg else 'OFF'}**")
    elif text == 'sub slay et':
        sub_slay_et = not sub_slay_et
        await event.reply(f"⚔️ Sub-Mode Slay Eternal: **{'ON' if sub_slay_et else 'OFF'}**")
        
    elif text == '.stats':
        recent_names = stats['slayed_names'][-15:]
        names_str = ", ".join(recent_names) if recent_names else "None yet"
        
        msg = (
            "📊 **SESSION ENGAGEMENT STATS & SETTINGS** 📊\n\n"
            f"⚙️ **Current Settings:**\n"
            f"  ├ Mode: **{combat_mode.upper()}**\n"
            f"  ├ Global Ignores: [NL:{'ON' if ignore_nl else 'OFF'}] [Leg:{'ON' if ignore_leg else 'OFF'}] [Et:{'ON' if ignore_et else 'OFF'}]\n"
            f"  └ Sub-Mode Slays: [NL:{'ON' if sub_slay_nl else 'OFF'}] [Leg:{'ON' if sub_slay_leg else 'OFF'}] [Et:{'ON' if sub_slay_et else 'OFF'}]\n\n"
            f"🗡️ **Characters Slayed:**\n"
            f"  ├ Non-Legendary: {stats['nl']}\n"
            f"  ├ Legendary: {stats['leg']}\n"
            f"  └ Eternal: {stats['et']}\n\n"
            f"🧿 **Monsters Sealed:** {stats['monsters_sealed']}\n\n"
            f"📜 **Recent Characters Slayed:**\n{names_str}"
        )
        await event.reply(msg)

# --- MAIN GAME HANDLER ---
@client.on(events.NewMessage(chats=TARGET_BOT_ID))
@client.on(events.MessageEdited(chats=TARGET_BOT_ID))
async def game_handler(event):
    global last_g3_time, last_g4_time
    global is_running, combat_mode, stats
    global ignore_nl, ignore_leg, ignore_et
    global sub_slay_nl, sub_slay_leg, sub_slay_et
    
    if not is_running: return

    response_received_event.set()
    
    raw_text = event.raw_text or ""
    clean_text = unicodedata.normalize('NFKC', raw_text).lower()
    clean_text = clean_text.replace(" ", "").replace("-", "").replace("_", "").replace("\n", "")

    try:
        current_time = time.time()

        # ==========================================
        # GROUP 4: TRADE, CRATE & BEAST BYPASS
        # ==========================================
        if any(kw in clean_text for kw in ["mysterioustrader", "auracrate", "tailedbeast", "wanderunseen"]):
            if current_time - last_g4_time < 3.0: return 
            last_g4_time = current_time 
            print("▶ Group 4: Bypass Trigger hit! Sending /explore...")
            await trigger_explore()
            return

        # ==========================================
        # GROUP 3: LOOT, DEFEAT & SKIPS
        # ==========================================
        elif any(kw in clean_text for kw in ["solved", "crate", "seal", "reward", "were", "smiles", "retreated"]):
            if current_time - last_g3_time < 3.0: return 
            last_g3_time = current_time 
            print("▶ Group 3: Loot/Skip Trigger hit. Sending /explore...")
            await trigger_explore()
            return

        # ==========================================
        # GROUP 1: UNIVERSAL IMAGE CHALLENGES
        # ==========================================
        elif any(kw in clean_text for kw in ["slot", "hit", "pokemon", "code", "captcha", "guess"]) and event.photo and event.message.buttons:
            print("▶ Group 1: Image Challenge Detected")
            photo_path = await event.download_media()
            ocr_text = await extract_text_from_image(photo_path)
            
            clicked = False
            clean_ocr = re.sub(r'(?i)(pick|pck|pic|piks|pcks:?|hit:?)[^a-z0-9]*', '', ocr_text).strip()
            if not clean_ocr: clean_ocr = ocr_text

            button_texts = [btn.text.lower() for row in event.message.buttons for btn in row if btn.text]
            if button_texts:
                matches = difflib.get_close_matches(clean_ocr, button_texts, n=1, cutoff=0.25)
                if matches:
                    for row_idx, row in enumerate(event.message.buttons):
                        for col_idx, btn in enumerate(row):
                            if btn.text and btn.text.lower() == matches[0]:
                                await asyncio.sleep(random.uniform(1.0, 2.0))
                                # 3-Attempt Retry Loop
                                for attempt in range(3):
                                    try:
                                        await event.message.click(row_idx, col_idx)
                                        clicked = True
                                        break
                                    except Exception as e:
                                        print(f"⚠️ CAPTCHA MATCH RETRY ({attempt+1}/3): {e}")
                                        await asyncio.sleep(1.0)
                                        
                                if not clicked:
                                    is_running = False
                                    if watchdog_task: watchdog_task.cancel()
                                    await client.send_message('me', f"🔴 **FAILSAFE:** Captcha match button failed after 3 retries. Script STOPPED.")
                                    return
                                break
                        if clicked: break

            if not clicked:
                match = re.search(r'\d', ocr_text) 
                if match:
                    t_num = int(match.group(0))
                    if 1 <= t_num <= 9:
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                        # 3-Attempt Retry Loop
                        for attempt in range(3):
                            try:
                                await event.message.click((t_num - 1) // 3, (t_num - 1) % 3)
                                clicked = True
                                break
                            except Exception as e:
                                print(f"⚠️ CAPTCHA GRID RETRY ({attempt+1}/3): {e}")
                                await asyncio.sleep(1.0)
                                
                        if not clicked:
                            is_running = False
                            if watchdog_task: watchdog_task.cancel()
                            await client.send_message('me', f"🔴 **FAILSAFE:** Captcha grid button failed after 3 retries. Script STOPPED.")
                            return

            if not clicked:
                is_running = False 
                if watchdog_task: watchdog_task.cancel()
                await client.send_message('me', "🔴 **FAILSAFE:** OCR failed to match anything. Script STOPPED.")
            return 

        # ==========================================
        # GROUP 2: COMBAT ENCOUNTERS
        # ==========================================
        elif any(kw in clean_text for kw in ["yourself", "trembles", "encountered", "character"]) and event.message.buttons:
            print(f"▶ Group 2: Combat Detected. Action Mode: [{combat_mode.upper()}]")
            
            char_name = "Unknown"
            name_match = re.search(r'Name:\s*([^\n]+)', raw_text)
            if name_match:
                char_name = name_match.group(1).strip()

            is_monster = "encountered" in clean_text or "yourself" in clean_text
            is_character = "character" in clean_text or "trembles" in clean_text
            
            is_nl = "rarity:nonlegendary" in clean_text
            is_leg = "rarity:legendary" in clean_text
            is_et = "rarity:eternal" in clean_text

            # --- MODE SKIPS ---
            if combat_mode == 'slay' and is_monster:
                print("⏭️ SLAY MODE: Saw a Monster. Skipping it!")
                await trigger_explore()
                return

            if combat_mode == 'subjugate' and is_character:
                exception_triggered = False
                if is_nl and sub_slay_nl: exception_triggered = True
                if is_leg and sub_slay_leg: exception_triggered = True
                if is_et and sub_slay_et: exception_triggered = True
                
                if not exception_triggered:
                    print("⏭️ SUBJUGATE MODE: Saw a Character. Skipping it!")
                    await trigger_explore()
                    return
                else:
                    print(f"⚠️ SUB EXCEPTION: Saw a target rarity ({char_name}). Engaging instead of skipping!")

            # --- GLOBAL RARITY SKIPS ---
            if is_character:
                if is_nl and ignore_nl:
                    print(f"⏭️ IGNORE FILTER: Skipping Non-Legendary ({char_name})!")
                    await trigger_explore()
                    return
                if is_leg and ignore_leg:
                    print(f"⏭️ IGNORE FILTER: Skipping Legendary ({char_name})!")
                    await trigger_explore()
                    return
                if is_et and ignore_et:
                    print(f"⏭️ IGNORE FILTER: Skipping Eternal ({char_name})!")
                    await trigger_explore()
                    return

            # --- STATS TRACKER ---
            if is_character:
                if is_nl: stats['nl'] += 1
                elif is_leg: stats['leg'] += 1
                elif is_et: stats['et'] += 1
                else: stats['unknown_char'] += 1
                stats['slayed_names'].append(char_name)
            elif is_monster:
                stats['monsters_sealed'] += 1

            # --- CLICK THE FIGHT BUTTON ---
            await asyncio.sleep(random.uniform(1.0, 2.0))
            clicked = False
            for row_idx, row in enumerate(event.message.buttons):
                for col_idx, btn in enumerate(row):
                    if not btn.text: continue
                    btn_lower = btn.text.lower()
                    
                    if any(word in btn_lower for word in ['slay', 'attack', 'subjugate', 'seal', 'subjucate', 'fight']):
                        print(f"⚔️ Engaging: '{char_name}'")
                        # 3-Attempt Retry Loop
                        for attempt in range(3):
                            try:
                                await event.message.click(row_idx, col_idx)
                                clicked = True
                                break
                            except Exception as e:
                                print(f"⚠️ COMBAT RETRY ({attempt+1}/3): {e}")
                                await asyncio.sleep(1.0)
                        break
                if clicked: break

            if not clicked:
                for attempt in range(3):
                    try:
                        await event.message.click(0, 0)
                        break
                    except Exception as e:
                        print(f"⚠️ COMBAT FALLBACK RETRY ({attempt+1}/3): {e}")
                        await asyncio.sleep(1.0)

    # ==========================================
    # GLOBAL ERROR CATCHER
    # ==========================================
    except Exception as e:
        print(f"❌ Error handling message: {e}")

print("Userbot is running on Host (100% Offline OCR + Clean Terminal Output)...")
client.start()
client.run_until_disconnected()