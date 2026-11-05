import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_MAP = {
    "hi": os.path.join(BASE_DIR, "..", "fonts", "NotoSansDevanagari-Regular.ttf"),  # Hindi
    "mr": os.path.join(BASE_DIR, "..", "fonts", "NotoSansDevanagari-Regular.ttf"),  # Marathi
    "ne": os.path.join(BASE_DIR, "..", "fonts", "NotoSansDevanagari-Regular.ttf"),  # Nepali
    "ar": os.path.join(BASE_DIR, "..", "fonts", "NotoSansArabic-Regular.ttf"),      # Arabic
    "ur": os.path.join(BASE_DIR, "..", "fonts", "NotoSansArabic-Regular.ttf"),      # Urdu
    "bn": os.path.join(BASE_DIR, "..", "fonts", "NotoSansBengali-Regular.ttf"),     # Bengali
    "pa": os.path.join(BASE_DIR, "..", "fonts", "NotoSansGurmukhi-Regular.ttf"),    # Punjabi
    "kn": os.path.join(BASE_DIR, "..", "fonts", "NotoSansKannada-Regular.ttf"),     # Kannada
    "ta": os.path.join(BASE_DIR, "..", "fonts", "NotoSansTamil-Regular.ttf"),       # Tamil
    "te": os.path.join(BASE_DIR, "..", "fonts", "NotoSansTelugu-Regular.ttf"),      # Telugu
}

DEFAULT_FONT = os.path.join(BASE_DIR, "..", "fonts", "NotoSans-Regular.ttf")