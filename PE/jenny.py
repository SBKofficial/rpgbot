import io
from PIL import Image

import google.generativeai as genai

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

# =========================
# CONFIG
# =========================

BOT_TOKEN = "8341690614:AAEEzCkF7CJ5cHPH0K1cnLwpJclgeqvtlqM"

GEMINI_API_KEY = "AIzaSyDBOMQBHaOfiPL0h9T62tOkPlAbWOf_uXc"

genai.configure(
    api_key=GEMINI_API_KEY
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =========================
# GEMINI SOLVER
# =========================

async def solve_with_gemini(image_bytes):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    prompt = """
You are solving a Pokémon captcha.

The image contains:
- One Pokémon at the top
- Six numbered options below

Your task:
Identify which option matches the Pokémon shown at the top.

Rules:
- Match same Pokémon species
- Shiny and normal forms count as SAME Pokémon
- Reply ONLY with the option number
- Reply with only one digit from 1 to 6
"""

    response = model.generate_content([
        prompt,
        image
    ])

    answer = response.text.strip()

    return answer


# =========================
# START COMMAND
# =========================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Send a Jenny captcha image."
    )


# =========================
# PHOTO HANDLER
# =========================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    try:

        print("📩 Image received")

        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = (
            await file.download_as_bytearray()
        )

        print("🧠 Sending to Gemini")

        answer = await solve_with_gemini(
            image_bytes
        )

        print("✅ Gemini Answer:", answer)

        await update.message.reply_text(
            f"✅ Answer: {answer}"
        )

    except Exception as e:

        import traceback

        print(
            traceback.format_exc()
        )

        await update.message.reply_text(
            f"❌ Error:\n{str(e)}"
        )


# =========================
# MAIN
# =========================

def main():

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    print(
        "Gemini Jenny Solver Running..."
    )

    app.run_polling()


if __name__ == "__main__":
    main()
