import io
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

BOT_TOKEN = "8341690614:AAEEzCkF7CJ5cHPH0K1cnLwpJclgeqvtlqM"


async def solve_image(image_bytes):

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

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

    top_hash = imagehash.phash(
        top_crop
    )

    top_arr = np.array(
        top_crop.resize((64, 64))
    ).astype(np.float32)

    best_match = None
    best_score = -999999

    all_scores = {}

    for i in range(1, 7):

        option_crop = crop_region(
            centers[i][0],
            centers[i][1]
        )

        # =========================
        # HASH SCORE
        # =========================

        option_hash = imagehash.phash(
            option_crop
        )

        hash_diff = (
            top_hash - option_hash
        )

        hash_score = max(
            0,
            64 - hash_diff
        )

        # =========================
        # PIXEL DIFFERENCE
        # =========================

        option_arr = np.array(
            option_crop.resize((64, 64))
        ).astype(np.float32)

        pixel_diff = np.mean(
            np.abs(top_arr - option_arr)
        )

        pixel_score = (
            255 - pixel_diff
        )

        # =========================
        # COLOR DIFFERENCE
        # =========================

        top_mean = np.mean(
            top_arr,
            axis=(0, 1)
        )

        option_mean = np.mean(
            option_arr,
            axis=(0, 1)
        )

        color_diff = np.mean(
            np.abs(
                top_mean - option_mean
            )
        )

        color_score = (
            255 - color_diff
        )

        # =========================
        # FINAL SCORE
        # =========================

        final_score = (
            (hash_score * 4.0)
            +
            (pixel_score * 1.5)
            +
            (color_score * 1.0)
        )

        all_scores[i] = {
            "hash": round(hash_score, 2),
            "pixel": round(pixel_score, 2),
            "color": round(color_score, 2),
            "final": round(final_score, 2)
        }

        print(
            f"Option {i} | "
            f"HASH={hash_score} | "
            f"PIXEL={pixel_score:.2f} | "
            f"COLOR={color_score:.2f} | "
            f"FINAL={final_score:.2f}"
        )

        if final_score > best_score:

            best_score = final_score
            best_match = i

    return {
        "answer": best_match,
        "score": round(best_score, 2),
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

        print("📩 Image received")

        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = (
            await file.download_as_bytearray()
        )

        print("🧠 Solving image")

        result = await solve_image(
            image_bytes
        )

        print("✅ Done")

        text = (
            f"✅ Answer: {result['answer']}\n"
            f"📊 Score: {result['score']}\n\n"
        )

        for k, v in result[
            "all_scores"
        ].items():

            text += (
                f"Option {k}\n"
                f"HASH: {v['hash']}\n"
                f"PIXEL: {v['pixel']}\n"
                f"COLOR: {v['color']}\n"
                f"FINAL: {v['final']}\n\n"
            )

        await update.message.reply_text(
            text
        )

    except Exception as e:

        import traceback

        print(
            traceback.format_exc()
        )

        await update.message.reply_text(
            f"❌ Error:\n{str(e)}"
        )


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
        "Pure PIL Jenny Solver Running..."
    )

    app.run_polling()


if __name__ == "__main__":
    main()
