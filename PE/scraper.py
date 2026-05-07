import requests
from PIL import Image
import imagehash
import json
import io
import concurrent.futures

TOTAL_POKEMON = 251

URLS = {
    "retro_normal": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/{}.png",
    "retro_shiny": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/shiny/{}.png",
    "modern_normal": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png",
    "modern_shiny": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/shiny/{}.png"
}

# 1. Use a Session to keep the connection alive (massive speed boost)
session = requests.Session()

def process_api_image(res_content):
    img = Image.open(io.BytesIO(res_content)).convert("RGBA")
    bbox = img.split()[-1].getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (64, 64), (255, 255, 255))
    bg.paste(img, (0, 0), img)
    return str(imagehash.phash(bg))

def process_pokemon(poke_id):
    """Fetches all 4 variations for a single Pokemon ID."""
    local_db = {"retro": {}, "modern": {}}
    
    def fetch_and_hash(url_template, category):
        try:
            res = session.get(url_template.format(poke_id), timeout=5)
            if res.status_code == 200:
                img_hash = process_api_image(res.content)
                local_db[category][img_hash] = poke_id
        except:
            pass # Ignore missing sprites or timeouts
            
    fetch_and_hash(URLS["retro_normal"], "retro")
    fetch_and_hash(URLS["retro_shiny"], "retro")
    fetch_and_hash(URLS["modern_normal"], "modern")
    fetch_and_hash(URLS["modern_shiny"], "modern")
    
    return local_db

if name == "main":
    print(f"Building Database with Multithreading... Hang tight!")
    
    final_database = {"retro": {}, "modern": {}}
    
    # 2. Fire off 20 requests at the exact same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Map the function to all IDs
        results = executor.map(process_pokemon, range(1, TOTAL_POKEMON + 1))
        
        # 3. Collect the threads as they finish and merge them into the master dictionary
        for i, result in enumerate(results, 1):
            final_database["retro"].update(result["retro"])
            final_database["modern"].update(result["modern"])
            
            if i % 25 == 0:
                print(f"Processed {i}/{TOTAL_POKEMON} Pokemon...")

    with open("database.json", "w") as f:
        json.dump(final_database, f)

    print("Done! Database built at lightning speed.")