import os
import json
from PIL import Image
import imagehash

# Paths
CARDS_FOLDER = 'official_cards'
CACHE_FILE = 'shinobi_cache.json'

def preload_cache():
    # Load the existing cache so we don't overwrite what you already have
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
        print(f"📦 Loaded existing cache with {len(cache)} entries.")
    else:
        cache = {}

    if not os.path.exists(CARDS_FOLDER):
        print(f"⚠️ Folder '{CARDS_FOLDER}' not found. Please create it and add images.")
        return

    added_count = 0

    # Loop through every downloaded image in the folder
    for filename in os.listdir(CARDS_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            # The name of the shinobi is just the filename without the .jpg part
            shinobi_name = os.path.splitext(filename)[0].title()
            image_path = os.path.join(CARDS_FOLDER, filename)
            
            try:
                # Generate the visual fingerprint
                img = Image.open(image_path)
                img_hash = str(imagehash.phash(img))
                
                # Add it to the cache dictionary
                cache[img_hash] = shinobi_name
                added_count += 1
                print(f"✅ Injected: {shinobi_name} (Hash: {img_hash})")
                
                img.close()
            except Exception as e:
                print(f"⚠️ Failed to process {filename}: {e}")

    # Save the updated cache back to the JSON file
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)
        
    print(f"\n🎉 Successfully injected {added_count} new shinobi into the cache!")
    print("You can now safely start your main catcher bot.")

if __name__ == "__main__":
    preload_cache()

