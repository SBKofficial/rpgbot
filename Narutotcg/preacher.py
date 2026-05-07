import os
import io
import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
import imagehash
from urllib.parse import urljoin

# --- CONFIGURATION ---
# Replace this with the actual URL of the card gallery or wiki
TARGET_URL = "https://www.narutotcgmythos.com/card-gallery" 
SAVE_FOLDER = "official_cards"
CACHE_FILE = "shinobi_cache.json"

def scrape_and_save_cards():
    # Setup folder and cache
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}

    print(f"🌍 Fetching webpage: {TARGET_URL}...")
    try:
        # Pretend to be a regular browser to avoid getting blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(TARGET_URL, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to load page: {e}")
        return

    # Parse the website's HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all image tags on the page
    images = soup.find_all('img')
    print(f"🔍 Found {len(images)} images on the page. Processing...")

    added_count = 0

    for img_tag in images:
        img_url = img_tag.get('src')
        if not img_url:
            continue
            
        # Ensure the URL is complete (handles relative links like '/images/card.jpg')
        img_url = urljoin(TARGET_URL, img_url)
        
        # Try to get the shinobi name from the image's 'alt' text, or fallback to the filename
        alt_text = img_tag.get('alt', '').strip()
        if alt_text:
            shinobi_name = alt_text.title()
        else:
            filename = img_url.split('/')[-1].split('.')[0]
            shinobi_name = filename.replace('_', ' ').replace('-', ' ').title()
            
        # Skip small icons or layout images (optional filter)
        if len(shinobi_name) < 3 or "logo" in shinobi_name.lower():
            continue

        try:
            print(f"⏳ Downloading: {shinobi_name}...")
            img_response = requests.get(img_url, headers=headers, timeout=10)
            img_response.raise_for_status()
            
            # Open image for hashing
            img_data = img_response.content
            img_obj = Image.open(io.BytesIO(img_data))
            
            # 1. Save the actual physical file
            safe_filename = "".join(c for c in shinobi_name if c.isalnum() or c in " _-").rstrip()
            save_path = os.path.join(SAVE_FOLDER, f"{safe_filename}.jpg")
            
            # Convert to RGB to ensure smooth JPG saving
            if img_obj.mode in ("RGBA", "P"):
                img_obj = img_obj.convert("RGB")
                
            img_obj.save(save_path, "JPEG")
            
            # 2. Calculate pHash and add to cache
            img_hash = str(imagehash.phash(img_obj))
            cache[img_hash] = shinobi_name
            
            added_count += 1
            print(f"✅ Saved & Cached: {shinobi_name} (Hash: {img_hash})")
            
            img_obj.close()
            
        except Exception as e:
            print(f"⚠️ Failed to download/process {shinobi_name}: {e}")

    # Save the updated memory bank
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)
        
    print(f"\n🎉 Successfully downloaded and cached {added_count} cards!")
    print(f"📁 Images saved to: /{SAVE_FOLDER}/")
    print(f"💾 Hashes saved to: {CACHE_FILE}")

if __name__ == "__main__":
    scrape_and_save_cards()

