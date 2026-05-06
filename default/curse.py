import os
from telethon import TelegramClient, events
from PIL import Image
import pytesseract

# --- CONFIGURATION ---
API_ID = 1747534 # Replace with your API ID
API_HASH = '5a2684512006853f2e48aca9652d83ea' # Replace with your API Hash

# Target specifics
GAME_BOT_USERNAME = 'Naruto_tcg_bot'
TARGET_GROUP = -1003923986174

# Initialize the client
client = TelegramClient('shinobi_userbot', API_ID, API_HASH)

# Listen only in the specific group, and only for messages from the bot
@client.on(events.NewMessage(chats=TARGET_GROUP, from_users=GAME_BOT_USERNAME))
async def shinobi_catcher(event):
    
    # Trigger regardless of rarity (ignores the emoji at the start)
    if 'ᴀ sʜɪɴᴏʙɪ ʜᴀs ᴀᴘᴘᴇᴀʀᴇᴅ!' in event.raw_text:
        print("🚨 Shinobi Detected! Downloading image...")
        
        # Check for the image
        if event.message.media:
            image_path = await event.download_media(file="shinobi_spawn.jpg")
            
            try:
                # Read text from image
                img = Image.open(image_path)
                extracted_text = pytesseract.image_to_string(img)
                
                # Split text into lines and clean out empty ones
                lines = extracted_text.split('\n')
                valid_lines = [line.strip() for line in lines if line.strip()]
                
                if valid_lines:
                    # Grab the very first line (the name at the top of the card)
                    shinobi_name = valid_lines[0]
                    
                    # Clean out weird symbols, leaving only letters and spaces
                    shinobi_name = "".join(c for c in shinobi_name if c.isalpha() or c.isspace())
                    
                    # Send ONLY the /catch command to the group
                    catch_command = f'/catch {shinobi_name.title()}'
                    await client.send_message(TARGET_GROUP, catch_command)
                    
                    print(f"✅ Sent: {catch_command}")
                else:
                    print("❌ Could not read any text from the image.")
            
            except Exception as e:
                print(f"⚠️ Error reading image: {e}")
            
            finally:
                # Clean up the image file
                if os.path.exists(image_path):
                    os.remove(image_path)

print(f"🥷 Shinobi Auto-Catcher is running in group {TARGET_GROUP}...")
with client:
    client.run_until_disconnected()