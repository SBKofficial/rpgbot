import os
import requests
import time

# Create directories if they don't exist
os.makedirs('retro_sprites', exist_ok=True)
os.makedirs('modern_sprites', exist_ok=True)

# The raw GitHub URLs where PokeAPI stores the actual image files
# Using Gen 2 (Gold) for the retro sprites to match the white backgrounds
RETRO_URL_TEMPLATE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/versions/generation-ii/gold/{}.png"
# Using the default modern sprites (Gen 4/5 style)
MODERN_URL_TEMPLATE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png"

# Gen 1 and Gen 2 total 251 Pokemon. Change this if the bot uses higher generations in the retro slot.
TOTAL_POKEMON = 251 

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

print(f"Starting download of {TOTAL_POKEMON} Pokemon sprites...")

for poke_id in range(1, TOTAL_POKEMON + 1):
    retro_path = os.path.join('retro_sprites', f'{poke_id}.png')
    modern_path = os.path.join('modern_sprites', f'{poke_id}.png')

    # Download Retro Sprite
    if not os.path.exists(retro_path):
        success = download_image(RETRO_URL_TEMPLATE.format(poke_id), retro_path)
        if not success:
            print(f"Failed to download retro sprite for ID {poke_id}")

    # Download Modern Sprite
    if not os.path.exists(modern_path):
        success = download_image(MODERN_URL_TEMPLATE.format(poke_id), modern_path)
        if not success:
            print(f"Failed to download modern sprite for ID {poke_id}")

    # Print progress every 10 downloads
    if poke_id % 10 == 0:
        print(f"Downloaded {poke_id} / {TOTAL_POKEMON}...")

    # Small sleep to avoid getting IP blocked by GitHub
    time.sleep(0.1) 

print("Download complete! Your local databases are ready.")