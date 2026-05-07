import telebot
from PIL import Image
import imagehash
import json
import io

TOKEN = '8341690614:AAEEzCkF7CJ5cHPH0K1cnLwpJclgeqvtlqM'
bot = telebot.TeleBot(TOKEN)

# 1. Load the offline dictionary into memory
print("Loading hash database...")
with open("database.json", "r") as f:
    db_data = json.load(f)

# Convert string hashes back to imagehash objects for math comparison
RETRO_DB = {imagehash.hex_to_hash(h): pid for h, pid in db_data["retro"].items()}
MODERN_DB = {imagehash.hex_to_hash(h): pid for h, pid in db_data["modern"].items()}
print("Database loaded. Bot is ready.")

def find_closest_id(target_hash, target_db):
    """Finds the closest matching Pokedex ID in the given database."""
    best_id = None
    lowest_diff = float('inf')
    
    for db_hash, poke_id in target_db.items():
        diff = target_hash - db_hash
        if diff < lowest_diff:
            lowest_diff = diff
            best_id = poke_id
            
    return best_id

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file)).convert('RGB')
        w, h = img.size

        # Grid centers based on the UI
        centers = {
            'top': (w * 0.50, h * 0.14),
            1: (w * 0.18, h * 0.48),
            2: (w * 0.50, h * 0.48),
            3: (w * 0.82, h * 0.48),
            4: (w * 0.18, h * 0.83),
            5: (w * 0.50, h * 0.83),
            6: (w * 0.82, h * 0.83)
        }
        
        crop_radius = int(w * 0.10)
        def get_hash(center):
            cx, cy = center
            crop = img.crop((cx - crop_radius, cy - crop_radius, cx + crop_radius, cy + crop_radius))
            return imagehash.phash(crop)

        # 2. Identify the Target Pokemon (Using the Retro dictionary)
        top_hash = get_hash(centers['top'])
        target_poke_id = find_closest_id(top_hash, RETRO_DB)
        
        # 3. Scan the 6 options (Using the Modern dictionary)
        best_match = None
        for i in range(1, 7):
            opt_hash = get_hash(centers[i])
            opt_poke_id = find_closest_id(opt_hash, MODERN_DB)
            
            # If the option's ID matches the target's ID, we have our winner
            if opt_poke_id == target_poke_id:
                best_match = i
                break
                
        if best_match:
            bot.reply_to(message, str(best_match))
        else:
            bot.reply_to(message, "Match not found in database.")

    except Exception as e:
        bot.reply_to(message, f"Error: {e}")

bot.polling()

