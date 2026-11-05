from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def draw_text(frame, text, position, font_path, size=32):
    """Render Unicode text on an OpenCV BGR frame using Pillow."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    obj = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size)
    obj.text(position, text, font=font, fill=(255, 255, 0))

    bgr_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return bgr_frame


def draw_text_with_bg(frame, text, position, font_path, size=32,
                      text_color=(255, 255, 255), bg_color=(0, 0, 0), padding=6):
    """Render Unicode text with a semi-transparent background box."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    font = ImageFont.truetype(font_path, size)

    # Measure text bounding box
    tmp = ImageDraw.Draw(image)
    bbox = tmp.textbbox(position, text, font=font)   # (left, top, right, bottom)
    x0 = bbox[0] - padding
    y0 = bbox[1] - padding
    x1 = bbox[2] + padding
    y1 = bbox[3] + padding

    # Draw semi-transparent background
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x0, y0, x1, y1], fill=(*bg_color, 160))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    obj = ImageDraw.Draw(image)
    obj.text(position, text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_confidence_bar(frame, confidence: float, position, width=200, height=12):
    """Draw a horizontal confidence bar on the frame.

    confidence : float in [0, 1]
    position   : (x, y) top-left corner of the bar
    """
    x, y = position
    # Background track
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (60, 60, 60), -1)
    # Filled portion — colour shifts green → yellow → red with confidence
    fill_w = int(width * confidence)
    if confidence > 0.75:
        bar_color = (0, 220, 100)   # green
    elif confidence > 0.5:
        bar_color = (0, 200, 230)   # amber
    else:
        bar_color = (60, 80, 220)   # red-ish
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (120, 120, 120), 1)
    # Percentage label
    pct_text = f"{int(confidence * 100)}%"
    cv2.putText(frame, pct_text, (x + width + 6, y + height - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame

