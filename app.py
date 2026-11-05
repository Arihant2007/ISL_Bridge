"""ISL Bridge: a Streamlit starter for camera-based sign recognition.

Recognition is deliberately enabled only when a compatible trained model is
provided. Landmark extraction alone cannot identify an ISL sign reliably.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "isl_classifier.joblib"

st.set_page_config(page_title="ISL Bridge", page_icon="🤟", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f1f6ff; padding: 1rem; border-radius: 14px;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#124cce,#3977f6); color:white;}
    .hero h1 {color:white; margin:0 0 .35rem 0;} .hero p {margin:0; opacity:.93;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🤟 ISL Bridge</h1><p>Sign Language to Text & Speech · A camera-first accessibility demo</p></div>',
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.header("How to use")
    st.markdown("1. Allow camera access or upload a frame.\n2. Capture a single, well-lit sign.\n3. Review the prediction and optionally speak it aloud.")
    st.caption("This prototype processes one frame at a time. Continuous sentence translation needs a sequence model trained on labelled ISL videos.")
    st.divider()
    st.caption("Your camera image is processed in this app session and is not saved by ISL Bridge.")

camera_col, result_col = st.columns([1.15, 0.85], gap="large")
with camera_col:
    st.subheader("Capture a sign")
    source = st.radio("Input source", ["Camera", "Upload image"], horizontal=True, label_visibility="collapsed")
    image = st.camera_input("Take a photo of a sign", help="Keep your hand fully in frame with a plain, well-lit background.") if source == "Camera" else st.file_uploader("Choose a hand-sign image", type=["jpg", "jpeg", "png"])

with result_col:
    st.subheader("Translation")
    if image:
        st.image(image, caption="Captured frame", use_container_width=True)
    if MODEL_PATH.exists():
        try:
            import joblib
            import mediapipe as mp
            import numpy as np
            from PIL import Image

            model = joblib.load(MODEL_PATH)
            rgb = np.asarray(Image.open(image).convert("RGB")) if image else None
            if rgb is None:
                st.info("Capture a frame to see a prediction.")
            else:
                with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.55) as hands:
                    result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    st.warning("No hand found. Try better lighting and keep your hand centered.")
                else:
                    # Expected feature format is documented in README; pad/truncate defensively.
                    features = np.zeros((1, 126), dtype=np.float32)
                    for hand_i, hand in enumerate(result.multi_hand_landmarks[:2]):
                        vals = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32).flatten()
                        features[0, hand_i * 63 : (hand_i + 1) * 63] = vals
                    prediction = str(model.predict(features)[0])
                    confidence = float(max(model.predict_proba(features)[0])) if hasattr(model, "predict_proba") else None
                    st.success(prediction)
                    if confidence is not None:
                        st.metric("Model confidence", f"{confidence:.0%}")
                    st.markdown(f'<div style="padding:1rem;border-radius:12px;background:#eef4ff;font-size:1.35rem"><b>{prediction}</b></div>', unsafe_allow_html=True)
                    st.components.v1.html(
                        """<button onclick="const t=window.parent.document.querySelector('[data-testid=stMarkdownContainer]'); const s=new SpeechSynthesisUtterance(%s); s.lang='en-IN'; window.speechSynthesis.speak(s);" style="padding:10px 16px;border:0;border-radius:10px;background:#2463eb;color:white;cursor:pointer">🔊 Speak translation</button>""" % json.dumps(prediction),
                        height=52,
                    )
        except Exception as exc:
            st.error("The model could not process this frame. Check that the model and dependencies match the documented format.")
            st.caption(f"Details: {exc}")
    elif image:
        st.warning("No trained ISL model is installed yet.")
        st.markdown("The camera capture is working, but accurate translation requires a labelled ISL dataset and a model trained for its vocabulary. This demo will not guess a sign.")
        st.code("models/isl_classifier.joblib", language="text")
    else:
        st.info("Capture an image to get started.")

st.divider()
st.subheader("Project pipeline")
cols = st.columns(5)
for col, (n, title) in zip(cols, [("01", "Camera"), ("02", "Hand landmarks"), ("03", "Gesture model"), ("04", "Text"), ("05", "Speech")]):
    col.markdown(f"**{n} · {title}**")
st.caption("Current scope: single-frame demo. A production translator should add temporal landmarks, signer-independent evaluation, and a validated ISL vocabulary.")

