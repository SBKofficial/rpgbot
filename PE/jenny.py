import io
import cv2
import imagehash
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

    # =========================
    # TOP IMAGE
    # =========================

    top_crop = crop_region(
        centers['top'][0],
        centers['top'][1]
    )

    top_gray = cv2.cvtColor(
        np.array(top_crop),
        cv2.COLOR_RGB2GRAY
    )

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(
        top_gray,
        None
    )

    top_hash = imagehash.phash(top_crop)

    top_hist = cv2.calcHist(
        [cv2.cvtColor(np.array(top_crop), cv2.COLOR_RGB2HSV)],
        [0, 1],
        None,
        [50, 60],
        [0, 180, 0, 256]
    )

    cv2.normalize(top_hist, top_hist)

    best_match = None
    best_final_score = -999999

    all_scores = {}

    bf = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    for i in range(1, 7):

        option_crop = crop_region(
            centers[i][0],
            centers[i][1]
        )

        # =========================
        # ORB SCORE
        # =========================

        option_gray = cv2.cvtColor(
            np.array(option_crop),
            cv2.COLOR_RGB2GRAY
        )

        kp2, des2 = orb.detectAndCompute(
            option_gray,
            None
        )

        orb_score = 0

        if des1 is not None and des2 is not None:

            matches = bf.match(des1, des2)

            good_matches = [
                m for m in matches
                if m.distance < 50
            ]

            orb_score = len(good_matches)

        # =========================
        # HASH SCORE
        # =========================

        option_hash = imagehash.phash(option_crop)

        hash_diff = top_hash - option_hash

        hash_score = max(0, 64 - hash_diff)

        # =========================
        # COLOR HISTOGRAM SCORE
        # =========================

        option_hist = cv2.calcHist(
            [cv2.cvtColor(np.array(option_crop), cv2.COLOR_RGB2HSV)],
            [0, 1],
            None,
            [50, 60],
            [0, 180, 0, 256]
        )

        cv2.normalize(option_hist, option_hist)

        hist_score = cv2.compareHist(
            top_hist,
            option_hist,
            cv2.HISTCMP_CORREL
        )

        hist_score = max(0, hist_score)

        # =========================
        # FINAL HYBRID SCORE
        # =========================

        final_score = (
            (orb_score * 5.0) +
            (hash_score * 2.0) +
            (hist_score * 20.0)
        )

        all_scores[i] = {
            "orb": round(orb_score, 2),
            "hash": round(hash_score, 2),
            "hist": round(hist_score, 2),
            "final": round(final_score, 2)
        }

        if final_score > best_final_score:
            best_final_score = final_score
            best_match = i

    SAFE_THRESHOLD = 40

    return {
        "answer": best_match,
        "safe": best_final_score >= SAFE_THRESHOLD,
        "score": round(best_final_score, 2),
        "all_scores": all_scores
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
            f"📊 Final Score: {result['score']}\n"
            f"🔐 Safe: {result['safe']}\n\n"
        )

        for k, v in result["all_scores"].items():

            text += (
                f"Option {k}\n"
                f"ORB: {v['orb']}\n"
                f"HASH: {v['hash']}\n"
                f"HIST: {v['hist']}\n"
                f"FINAL: {v['final']}\n\n"
            )

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

    print("Hybrid Jenny Solver Running...")

    app.run_polling()


if __name__ == "__main__":
    main()
