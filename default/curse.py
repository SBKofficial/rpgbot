from telethon.sync import TelegramClient
from telethon.sessions import StringSession

# Put your API_ID and API_HASH here
API_ID = 1747534 # Replace with your API ID
API_HASH = '5a2684512006853f2e48aca9652d83ea' # Replace with your API Hash

print("Starting session generator...")
with TelegramClient(StringSession(), API_ID, API_HASH) as client:
    print("\n--- HERE IS YOUR STRING SESSION ---")
    print(client.session.save())
    print("-----------------------------------\n")
    print("Copy the long string of text above and keep it secret!")