import io
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


def get_color_signature(img, cx, cy, radius):
    crop = img.crop((
        int(cx - radius),
        int(cy - radius),
        int(cx + radius),
        int(cy + radius)
    ))

    colors = crop.getcolors(crop.size[0] * crop.size[1])

    if not colors:
        return {}

    colors.sort(reverse=True, key=lambda x: x[0])

    bg_color = colors[0][1]

    signature = {}

    for count, color in colors:
        if sum(abs(a - b) for a, b in zip(color, bg_color)) < 40:
            continue

        q_color = (
            color[0] // 16,
            color[1] // 16,
            color[2] // 16
        )

        signature[q_color] = signature.get(q_color, 0) + count

    return signature


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

    top_sig = get_color_signature(
        img,
        centers['top'][0],
        centers['top'][1],
        crop_radius
    )

    best_match = None
    lowest_penalty = float('inf')

    penalties = {}

    for i in range(1, 7):
        opt_sig = get_color_signature(
            img,
            centers[i][0],
            centers[i][1],
            crop_radius
        )

        all_colors = set(top_sig.keys()).union(set(opt_sig.keys()))

        penalty = sum(
            abs(top_sig.get(c, 0) - opt_sig.get(c, 0))
            for c in all_colors
        )

        penalties[i] = penalty

        if penalty < lowest_penalty:
            lowest_penalty = penalty
            best_match = i

    CONFIDENCE_LIMIT = 5000

    return {
        "answer": best_match,
        "penalty": lowest_penalty,
        "safe": lowest_penalty <= CONFIDENCE_LIMIT,
        "all_penalties": penalties
    }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me a Jenny captcha image."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]

    file = await photo.get_file()

    image_bytes = await file.download_as_bytearray()

    result = await solve_image(image_bytes)

    text = (
        f"✅ Best Match: {result['answer']}\n"
        f"📉 Penalty: {result['penalty']}\n"
        f"🔐 Safe: {result['safe']}\n\n"
        f"📊 All Penalties:\n"
    )

    for k, v in result["all_penalties"].items():
        text += f"{k}: {v}\n"

    await update.message.reply_text(text)


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(
        MessageHandler(filters.PHOTO, handle_photo)
    )

    print("Bot running...")

    app.run_polling()


if __name__ == "__main__":
    main()
