import cv2
import mediapipe as mp
import os
from predict import predict_sign_with_confidence
from text_builder import TextAnalyzer
from speak import speak
from deep_translator import GoogleTranslator
from translate import translate
from config import FONT_MAP, DEFAULT_FONT
from utils import draw_text, draw_confidence_bar

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

# Landmark drawing spec — thin cyan circles
_LANDMARK_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 220), thickness=1, circle_radius=2)
_CONNECTION_SPEC = mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=1)


def main():

    text = TextAnalyzer()

    choice = input("""Camera input or Video file ?
                Choose 1 for camera 
                       2 for video file : """).strip()
    while True:

        if choice == "1":
            cap = cv2.VideoCapture(0)
            break

        elif choice == "2":
            video = input("Enter video file path : ").strip()

            while True:
                if not os.path.isfile(video):
                    video = input("Enter correct video file path : ")
                else:
                    break

            cap = cv2.VideoCapture(video)
            break

    speech = input("Do you want sentences spoken aloud when you press 'S'? (y/n) ").strip().lower()

    while True:
        if speech in ("y", "n"):
            break
        else:
            speech = input("Enter (y/n) : ").strip().lower()

    codes = GoogleTranslator().get_supported_languages(as_dict=True)
    lang = input("Enter the language you want to translate the text (and audio) into : ").lower().strip()

    while True:
        if lang in codes:
            language = "en" if lang == "english" else codes[lang]
            break
        else:
            lang = input("Enter a correct language : ")

    sentence_str = ""
    translated_str = ""
    last_sign = ""
    confidence = 0.0

    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            confirmed_sign = None
            if results.multi_hand_landmarks:
                # Draw hand skeleton overlay
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        _LANDMARK_SPEC, _CONNECTION_SPEC,
                    )

                dataL, dataR = [], []
                for i, hand in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[i].classification[0].label
                    for pt in hand.landmark:
                        coords = [pt.x, pt.y, pt.z]
                        (dataL if label.lower() == "left" else dataR).extend(coords)

                if not dataL:
                    data = [0] * 63 + dataR
                elif not dataR:
                    data = dataL + [0] * 63
                else:
                    data = dataL + dataR

                if len(data) == 126:
                    prediction, confidence = predict_sign_with_confidence(data)
                    prev_len = len(text.sentence)
                    confirmed_sign, _, _ = text.process(prediction)
                    if confirmed_sign is not None:
                        last_sign = confirmed_sign
                    if len(text.sentence) > prev_len and language != "en":
                        translated_str = translate(" ".join(text.sentence), language)

            word_str = "".join(text.word)
            sentence_str = " ".join(text.sentence)
            display_str = translated_str if language != "en" else sentence_str
            font_path = FONT_MAP.get(language, DEFAULT_FONT)

            # ── HUD overlay ───────────────────────────────────────────
            # Semi-transparent dark banner at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 195), (10, 10, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Sign label
            cv2.putText(frame, "SIGN", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 140), 1)
            cv2.putText(frame, last_sign if last_sign else "--", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 230, 255), 2)

            # Confidence bar (how sure the model is)
            frame = draw_confidence_bar(frame, confidence, (10, 70), width=180, height=10)
            cv2.putText(frame, "CONF", (196, 79),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1)

            # Hold-progress bar (how close to confirming the sign)
            hold_w = int(180 * text.hold_progress)
            cv2.rectangle(frame, (10, 86), (190, 94), (40, 40, 40), -1)
            if hold_w > 0:
                cv2.rectangle(frame, (10, 86), (10 + hold_w, 94), (60, 220, 60), -1)
            cv2.rectangle(frame, (10, 86), (190, 94), (90, 90, 90), 1)
            cv2.putText(frame, "HOLD", (196, 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 150), 1)

            # Current word being spelled
            cv2.putText(frame, "WORD", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 140), 1)
            cv2.putText(frame, word_str if word_str else "--", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (80, 255, 180), 2)

            # Sentence (Unicode via Pillow)
            frame = draw_text(frame, f"Sentence : {display_str}",
                              (10, 162), font_path, 26)

            # Key-hint bar at the very bottom
            hint = "  [S] Speak   [R] Reset word   [Esc] Quit"
            cv2.putText(frame, hint,
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 100), 1)

            cv2.imshow("ISL Bridge", frame)

        elif not success:
            break

        key = cv2.waitKey(1)
        if key == 27:  # Esc
            cap.release()
            cv2.destroyAllWindows()
            break
        elif key == ord("s"):
            if speech == "y" and sentence_str:
                speak(translated_str if language != "en" else sentence_str, language)
        elif key == ord("r"):
            # Reset current word (useful after a mistake)
            text.word = []


if __name__ == "__main__":
    main()
