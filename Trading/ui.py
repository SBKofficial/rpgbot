import requests

# 1. Put your demo bot token here
TOKEN = "8658706022:AAGFbilodTYp2G4oRKGxPO1BHv9A-O6Vtwk"

# 2. Put your personal Telegram User ID here
CHAT_ID = "7708811819"

# The expandable UI mockup with your specific name formatting
html_ui = """
🎯 <b>ℍ𝕆𝕋 & ℂ𝕆𝕃𝔻 𝔸ℝ𝔼ℕ𝔸</b>
Target Range: <code>1 - 1000</code>

<blockquote expandable>📜 <b>Guess History:</b>
<code>[ 100 ]</code> 🔼 Higher — <a href="tg://user?id=111">Alex</a>
<code>[ 500 ]</code> 🔽 Lower — <a href="tg://user?id=222">Sarah</a>
<code>[ 250 ]</code> 🔼 Higher — <a href="tg://user?id=333">『 𓄀 𝙔𝙖𝙢𝙞 𝙎𝙪𝙠𝙞𝙝𝙚𝙧𝙤 ✗ 』</a></blockquote>

⚡ <b>Latest:</b> <code>[ 415 ]</code> 🔼 Higher — <a href="tg://user?id=333">『 𓄀 𝙔𝙖𝙢𝙞 𝙎𝙪𝙠𝙞𝙝𝙚𝙧𝙤 ✗ 』</a>
"""

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": html_ui,
    "parse_mode": "HTML"
}

try:
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("Success! Check your Telegram DMs.")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed to connect: {e}")