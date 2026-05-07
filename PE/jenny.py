import telebot
from PIL import Image
import google.generativeai as genai
import io
import time

TOKEN = '8341690614:AAEEzCkF7CJ5cHPH0K1cnLwpJclgeqvtlqM'
bot = telebot.TeleBot(TOKEN)

# 1. Put all your free API keys in a list
API_KEYS = [
    'AIzaSyDBOMQBHaOfiPL0h9T62tOkPlAbWOf_uXc',
    'AIzaSyB7m791-hibcw5fpNkNlVLUhV48BEAhP5c'
]

# Track which key we are currently using
current_key_index = 0

def get_model():
    """Configures GenAI with the current active key."""
    genai.configure(api_key=API_KEYS[current_key_index])
    return genai.GenerativeModel('gemini-2.5-flash')

def extract_crops(img):
    w, h = img.size
    crop_radius = int(w * 0.12)
    
    centers = {
        'target': (w * 0.50, h * 0.14),
        1: (w * 0.18, h * 0.48),
        2: (w * 0.50, h * 0.48),
        3: (w * 0.82, h * 0.48),
        4: (w * 0.18, h * 0.83),
        5: (w * 0.50, h * 0.83),
        6: (w * 0.82, h * 0.83)
    }
    
    crops = {}
    for key, (cx, cy) in centers.items():
        box = (cx - crop_radius, cy - crop_radius, cx + crop_radius, cy + crop_radius)
        crops[key] = img.crop(box)
    return crops

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    global current_key_index 
    
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file)).convert('RGB')
        
        crops = extract_crops(img)
        
        api_images = []
        for key in ['target', 1, 2, 3, 4, 5, 6]:
            img_byte_arr = io.BytesIO()
            crops[key].save(img_byte_arr, format='JPEG')
            api_images.append({
                "mime_type": "image/jpeg",
                "data": img_byte_arr.getvalue()
            })
            
        prompt = (
            "You are a Pokemon expert. I am providing 7 cropped images of Pokemon. "
            "Image 1 is the Target. Images 2 through 7 are Options 1 through 6. "
            "Identify the species name of all 7 Pokemon. "
            "Return ONLY a comma-separated list of the 7 names in order."
        )
        content = [prompt] + api_images

        # 2. The Retry Loop
        max_retries = len(API_KEYS)
        attempts = 0
        response = None

        while attempts < max_retries:
            try:
                model = get_model()
                response = model.generate_content(content)
                break # If successful, break out of the retry loop
                
            except Exception as api_error:
                error_msg = str(api_error).lower()
                # Check if the error is a rate limit or quota issue
                if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                    print(f"Key {current_key_index + 1} hit a limit. Swapping keys...")
                    # Cycle to the next key in the list
                    current_key_index = (current_key_index + 1) % len(API_KEYS)
                    attempts += 1
                    time.sleep(1) # Brief pause before hammering the API again
                else:
                    # If it's a different error (like a bad image), stop trying
                    raise api_error
        
        if not response:
             bot.reply_to(message, "All API keys are currently rate limited. Try again in a minute.")
             return

        # 3. Parse the names 
        names = [name.strip().lower() for name in response.text.split(',')]
        
        if len(names) == 7:
            target_name = names[0]
            options_names = names[1:]
            
            if target_name in options_names:
                match_number = options_names.index(target_name) + 1 
                bot.reply_to(message, f"{match_number}")
            else:
                bot.reply_to(message, f"Target was {target_name.title()}, but couldn't find a match.")
        else:
            bot.reply_to(message, f"API returned weird formatting: {response.text}")

    except Exception as e:
        bot.reply_to(message, f"Error: {e}")

print("Multi-Key Auto-Cycling bot is running...")
bot.polling()

