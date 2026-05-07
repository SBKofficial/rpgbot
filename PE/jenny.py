import io
import cv2
import numpy as np

from PIL import Image

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

BOT_TOKEN = '8341690614:AAEEzCkF7CJ5cHPH0K1cnLwpJclgeqvtlqM'


async def solve_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = img.size

    centers = {
        'top': (w * 0.50, h * 0.16),

        1: (w * 0.18, h * 0.48),
        2: (w * 0.50, h * 0.48),
        3: (w * 0.82, h * 0.48),

        4: (w * 0.18, h * 0.83),
        5: (w * 0.50, h * 0.83),
        6: (w * 0.82, h * 0.83)
    }

    crop_radius = int(w * 0.10)

    def crop_region(cx, cy):
        return img.crop((
            int(cx - crop_radius),
            int(cy - crop_radius),
            int(cx + crop_radius),
            int(cy + crop_radius)
        ))

    # TOP IMAGE
    top_crop = crop_region(
        centers['top'][0],
        centers['top'][1]
    )

    top_cv = cv2.cvtColor(
        np.array(top_crop),
        cv2.COLOR_RGB2GRAY
    )

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(top_cv, None)

    if des1 is None:
        return {
            "answer": None,
            "safe": False,
            "score": 0,
            "all_scores": {}
        }

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    best_match = None
    best_score = -1

    scores = {}

    for i in range(1, 7):

        option_crop = crop_region(
            centers[i][0],
            centers[i][1]
        )

        option_cv = cv2.cvtColor(
            np.array(option_crop),
            cv2.COLOR_RGB2GRAY
        )

        kp2, des2 = orb.detectAndCompute(
            option_cv,
            None
        )

        if des2 is None:
            scores[i] = 0
            continue

        matches = bf.match(des1, des2)

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        # FILTER GOOD MATCHES
        good_matches = [
            m for m in matches
            if m.distance < 50
        ]

        score = len(good_matches)

        scores[i] = score

        if score > best_score:
            best_score = score
            best_match = i

    SAFE_THRESHOLD = 8

    return {
        "answer": best_match,
        "safe": best_score >= SAFE_THRESHOLD,
        "score": best_score,
        "all_scores": scores
    }


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "Send a Jenny captcha image."
    )


async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    try:
        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = await file.download_as_bytearray()

        result = await solve_image(image_bytes)

        text = (
            f"✅ Answer: {result['answer']}\n"
            f"📊 Score: {result['score']}\n"
            f"🔐 Safe: {result['safe']}\n\n"
            f"📈 Match Scores:\n"
        )

        for k, v in result["all_scores"].items():
            text += f"{k}: {v}\n"

        await update.message.reply_text(text)

    except Exception as e:
        await update.message.reply_text(
            f"❌ Error:\n{str(e)}"
        )


def main():
    app = Application.builder().token(
        BOT_TOKEN
    ).build()

    app.add_handler(
        CommandHandler("start", start)
    )

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo
        )
    )

    print("Jenny Solver Bot Running...")

    app.run_polling()


if __name__ == "__main__":
    main()
