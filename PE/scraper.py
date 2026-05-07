import os
import requests
from PIL import Image
import imagehash
import json
import io

# We will hash the first 251 Pokemon (Gen 1 & 2). Increase this if the bot uses higher gens.
TOTAL_POKEMON = 251

RETRO_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/{}.png"
MODERN_URL = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png"

database = {
    "retro": {},
    "modern": {}
}

print("Building offline hash database. This will take a minute...")

for poke_id in range(1, TOTAL_POKEMON + 1):
    try:
        # 1. Fetch Retro Sprite
        retro_res = requests.get(RETRO_URL.format(poke_id))
        if retro_res.status_code == 200:
            img = Image.open(io.BytesIO(retro_res.content)).convert("RGBA")
            # Create a white background to match the Telegram bot's top image
            bg = Image.new("RGBA", img.size, "WHITE")
            bg.paste(img, (0, 0), img)
            r_hash = str(imagehash.phash(bg.convert("RGB")))
            database["retro"][r_hash] = poke_id

        # 2. Fetch Modern Sprite
        modern_res = requests.get(MODERN_URL.format(poke_id))
        if modern_res.status_code == 200:
            img = Image.open(io.BytesIO(modern_res.content)).convert("RGBA")
            # Create a dark background to match the Telegram bot's option boxes
            bg = Image.new("RGBA", img.size, (43, 45, 49)) # Discord/Telegram dark grey
            bg.paste(img, (0, 0), img)
            m_hash = str(imagehash.phash(bg.convert("RGB")))
            database["modern"][m_hash] = poke_id

        if poke_id % 25 == 0:
            print(f"Hashed {poke_id}/{TOTAL_POKEMON}...")
            
    except Exception as e:
        print(f"Failed on ID {poke_id}: {e}")

# Save the fingerprints to a JSON file
with open("database.json", "w") as f:
    json.dump(database, f)

print("Success! database.json created. You can now run your main bot.")