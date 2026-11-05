import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request, send_file
from deep_translator import GoogleTranslator
import cv2
import mediapipe as mp
import numpy as np
import base64
from src.predict import predict_sign_with_confidence
from src.text_builder import TextAnalyzer
from src.translate import translate
import logging
from gtts import gTTS
import io

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
text = TextAnalyzer()
last_sign = ""
translated_str = ""
# Simple in-memory session history (list of dicts)
session_history: list[dict] = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/languages")
def get_all_supported_languages():
    # key - language name, value - language code
    name_code = GoogleTranslator().get_supported_languages(as_dict=True)
    return jsonify(name_code)


@app.route("/predict", methods=["POST"])
def get_prediction():
    global last_sign, translated_str, session_history

    data = request.get_json()
    language = data["language"]
    image_data = data["image"]
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    confirmed_sign = ""
    confidence = 0.0
    hold_progress = 0.0

    if results.multi_hand_landmarks:
        dataL = []
        dataR = []
        for i, hand in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            for point in hand.landmark:
                coords = [point.x, point.y, point.z]
                (dataL if label.lower() == "left" else dataR).extend(coords)

        if not dataL:
            landmark_data = [0] * 63 + dataR
        elif not dataR:
            landmark_data = dataL + [0] * 63
        else:
            landmark_data = dataL + dataR

        if len(landmark_data) == 126:
            prediction, confidence = predict_sign_with_confidence(landmark_data)
            prev_len = len(text.sentence)
            confirmed_sign_result, _, _ = text.process(prediction)
            hold_progress = text.hold_progress

            if confirmed_sign_result is not None:
                last_sign = confirmed_sign_result
                confirmed_sign = confirmed_sign_result

                # Record confirmed signs in session history
                if confirmed_sign_result not in ("SPACE", "BACKSPACE"):
                    # append to history with timestamp-like index
                    session_history.append({
                        "sign": confirmed_sign_result,
                        "confidence": round(confidence, 3),
                    })
                    # Keep history capped at 100 entries
                    if len(session_history) > 100:
                        session_history = session_history[-100:]

            if len(text.sentence) > prev_len:
                if language != "en":
                    translated_str = translate(" ".join(text.sentence), language)
                else:
                    translated_str = " ".join(text.sentence)

    return jsonify({
        "sign": last_sign,
        "word": "".join(text.word),
        "sentence": translated_str,
        "confidence": round(confidence, 3),
        "hold_progress": round(hold_progress, 3),
    })


@app.route("/translate", methods=["POST"])
def translation_when_lang_changed():
    global translated_str
    data = request.get_json()
    language_code = data["code"]

    if not text.sentence:
        return jsonify("")

    if language_code == "en":
        translated_str = " ".join(text.sentence)
    else:
        translated_str = translate(" ".join(text.sentence), language_code)

    return jsonify(translated_str)


@app.route("/reset")
def reset():
    global last_sign, translated_str, session_history
    last_sign = ""
    translated_str = ""
    text.word = []
    text.sentence = []
    text.reset_timer()
    session_history = []
    return jsonify("ok")


@app.route("/reset_word")
def reset_word():
    """Clear only the current in-progress word."""
    text.word = []
    text.reset_timer()
    return jsonify("ok")


@app.route("/history")
def get_history():
    return jsonify(session_history)


@app.route("/speak", methods=["POST"])
def speak():
    global translated_str
    data = request.get_json()
    if not translated_str:
        return jsonify(""), 400
    try:
        audio = gTTS(text=translated_str, lang=data["code"])
        loc = io.BytesIO()
        audio.write_to_fp(loc)
        loc.seek(0)
    except ValueError:
        return jsonify(""), 400
    return send_file(loc, mimetype="audio/mpeg")


if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=7860, debug=False)
