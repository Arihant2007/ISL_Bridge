"""
ISL Bridge — Streamlit App
Real-time Indian Sign Language Fingerspelling Recognition

Run:
    streamlit run streamlit_app.py
"""

import sys
import os
import io
import threading
import queue

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from deep_translator import GoogleTranslator
from gtts import gTTS

# ── Path setup so src/* imports work ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_sign_with_confidence
from text_builder import TextAnalyzer
from translate import translate

# ── MediaPipe ─────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

_mp_hands = mp.solutions.hands
_mp_drawing = mp.solutions.drawing_utils
_LANDMARK_SPEC = _mp_drawing.DrawingSpec(color=(0, 255, 220), thickness=1, circle_radius=2)
_CONNECTION_SPEC = _mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=1)

# ── RTC config (public STUN for local + deployed) ─────────────────────────────
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISL Bridge",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Header ── */
.isl-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 0; margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(59,130,246,0.2);
}
.isl-title { font-size: 2rem; font-weight: 700;
    background: linear-gradient(130deg,#fff 0%,#60a5fa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; }
.isl-sub { font-size: 0.72rem; color: #4b5563;
    letter-spacing: 2px; text-transform: uppercase; }
.badge { display: inline-block; font-size: 0.62rem; font-weight: 600;
    padding: 0.2rem 0.6rem; border-radius: 999px;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd; margin-left: 0.3rem; text-transform: uppercase; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(145deg,rgba(17,24,39,.7),rgba(8,13,26,.7));
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.card-label { font-size: 0.6rem; text-transform: uppercase; letter-spacing: 2px;
    color: #4b5563; font-family: 'JetBrains Mono', monospace; }
.sign-glyph { font-size: 3.5rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; color: #60a5fa;
    text-shadow: 0 0 24px rgba(96,165,250,.4); letter-spacing: 4px; }
.word-val { font-size: 1.5rem; font-weight: 600; color: #4ade80;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 3px; }
.sentence-val { font-size: 1.05rem; font-weight: 500; color: #f8fafc;
    line-height: 1.65; }

/* ── Progress bar override ── */
.stProgress > div > div > div > div { border-radius: 999px; }

/* ── History chips ── */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.4rem; }
.chip { display: inline-flex; flex-direction: column; align-items: center;
    padding: 0.25rem 0.55rem; border-radius: 8px;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    min-width: 36px; }
.chip-letter { font-size: 0.95rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; color: #93c5fd; }
.chip-pct { font-size: 0.55rem; color: #4b5563;
    font-family: 'JetBrains Mono', monospace; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: rgba(8,13,26,0.95) !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px; font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Session state init
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "text_analyzer": TextAnalyzer(),
        "last_sign":     "",
        "confidence":    0.0,
        "hold_progress": 0.0,
        "translated_str":"",
        "history":       [],     # list of {sign, confidence}
        "language_code": "en",
        "language_name": "english",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Shared result queue (thread-safe bridge from WebRTC thread → Streamlit) ───
if "result_queue" not in st.session_state:
    st.session_state["result_queue"] = queue.SimpleQueue()

result_q: queue.SimpleQueue = st.session_state["result_queue"]


# ══════════════════════════════════════════════════════════════════════════════
#  Language list (cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_languages() -> dict:
    """Returns {name: code} mapping from Google Translate."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


# ══════════════════════════════════════════════════════════════════════════════
#  WebRTC video-frame processor
# ══════════════════════════════════════════════════════════════════════════════
class ISLVideoProcessor:
    """Runs in a background thread for every incoming video frame."""

    def __init__(self):
        self._hands = _mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        confidence = 0.0
        prediction = None

        if results.multi_hand_landmarks:
            # Draw skeleton overlay
            for hand_lm in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    img, hand_lm,
                    _mp_hands.HAND_CONNECTIONS,
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

        # Push result to the main thread via queue
        result_q.put({"prediction": prediction, "confidence": confidence})

        # ── HUD drawn directly on the returned frame ──────────────────────
        h, w = img.shape[:2]

        # Dark banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        # Sign label (top-left)
        sign_text = st.session_state.get("last_sign", "") or "--"
        cv2.putText(img, sign_text, (12, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 230, 255), 3)

        # Confidence bar (right side)
        bar_x, bar_y, bar_w, bar_h = w - 160, 15, 140, 10
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (40, 40, 40), -1)
        fill = int(bar_w * confidence)
        bar_color = (0, 220, 100) if confidence > 0.75 else (0, 200, 230) if confidence > 0.5 else (60, 80, 220)
        if fill > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                          bar_color, -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (100, 100, 100), 1)
        cv2.putText(img, f"CONF {int(confidence*100)}%", (bar_x, bar_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # Hold-progress bar
        hp = st.session_state.get("hold_progress", 0.0)
        hp_fill = int(bar_w * hp)
        cv2.rectangle(img, (bar_x, bar_y + 18), (bar_x + bar_w, bar_y + 28),
                      (40, 40, 40), -1)
        if hp_fill > 0:
            cv2.rectangle(img, (bar_x, bar_y + 18), (bar_x + hp_fill, bar_y + 28),
                          (60, 220, 60), -1)
        cv2.rectangle(img, (bar_x, bar_y + 18), (bar_x + bar_w, bar_y + 28),
                      (100, 100, 100), 1)
        cv2.putText(img, f"HOLD {int(hp*100)}%", (bar_x, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _speak_audio(text_str: str, lang_code: str) -> bytes | None:
    """Generate MP3 bytes via gTTS (requires internet)."""
    try:
        tts = gTTS(text=text_str, lang=lang_code)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def _progress_html(value: float, color_class: str, label: str) -> str:
    pct = int(value * 100)
    color = "#3b82f6" if color_class == "conf" else "#22c55e"
    return f"""
    <div style="margin:4px 0">
      <span style="font-size:0.58rem;font-family:'JetBrains Mono',monospace;
                   color:#4b5563;letter-spacing:1px">{label}</span>
      <div style="height:8px;background:rgba(255,255,255,.05);border-radius:999px;
                  border:1px solid rgba(255,255,255,.07);overflow:hidden;margin-top:2px">
        <div style="width:{pct}%;height:100%;background:{color};
                    border-radius:999px;transition:width .15s ease-out"></div>
      </div>
    </div>
    """


def _chips_html(history: list) -> str:
    if not history:
        return '<span style="font-size:.7rem;color:#374151;font-family:JetBrains Mono,monospace">No signs yet</span>'
    chips = "".join(
        f'<div class="chip"><span class="chip-letter">{h["sign"]}</span>'
        f'<span class="chip-pct">{int(h["confidence"]*100)}%</span></div>'
        for h in history[-30:]  # show last 30
    )
    return f'<div class="chip-wrap">{chips}</div>'


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="isl-header">
  <div>
    <span class="isl-title">🤝 ISL Bridge</span>
    <div class="isl-sub">Indian Sign Language · Real-time Fingerspelling Recognition</div>
  </div>
  <div>
    <span class="badge">34 Signs</span>
    <span class="badge">LLM Word-Guess</span>
    <span class="badge">Multilingual</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — settings & controls
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Language selector
    lang_map = get_languages()          # {name: code}
    lang_names = ["english"] + [n for n in lang_map if n != "english"]
    sel_lang_name = st.selectbox(
        "🌐 Output Language",
        lang_names,
        index=lang_names.index(st.session_state["language_name"])
            if st.session_state["language_name"] in lang_names else 0,
        key="lang_select",
    )

    # Persist language change
    if sel_lang_name != st.session_state["language_name"]:
        st.session_state["language_name"] = sel_lang_name
        st.session_state["language_code"] = "en" if sel_lang_name == "english" else lang_map[sel_lang_name]
        # Re-translate existing sentence
        ta: TextAnalyzer = st.session_state["text_analyzer"]
        if ta.sentence:
            lc = st.session_state["language_code"]
            if lc == "en":
                st.session_state["translated_str"] = " ".join(ta.sentence)
            else:
                st.session_state["translated_str"] = translate(" ".join(ta.sentence), lc)

    st.markdown("---")
    st.markdown("### 🎛️ Controls")

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        reset_word_btn = st.button("✕ Word", use_container_width=True,
                                   help="Erase the current in-progress word")
    with col_b:
        reset_all_btn  = st.button("↺ Reset", use_container_width=True,
                                   help="Clear everything and start over")

    speak_btn = st.button("🔊 Speak Sentence", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. Allow webcam access when prompted  
2. Hold a sign **steady for 1.5 s** to confirm it  
3. Make a **fist** (SPACE gesture) to end a word  
4. The LLM will auto-correct the spelling  
5. Press **🔊 Speak** to hear the sentence  

**Keys in terminal app:**  
`S` → Speak · `R` → Reset word · `Esc` → Quit
""")

    st.markdown("---")
    st.caption("34 ISL fingerspelling signs · MediaPipe + PyTorch · Groq LLM · gTTS")


# ══════════════════════════════════════════════════════════════════════════════
#  RESET HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
if reset_all_btn:
    st.session_state["text_analyzer"] = TextAnalyzer()
    st.session_state["last_sign"]     = ""
    st.session_state["confidence"]    = 0.0
    st.session_state["hold_progress"] = 0.0
    st.session_state["translated_str"]= ""
    st.session_state["history"]       = []
    # Drain queue
    while not result_q.empty():
        try: result_q.get_nowait()
        except: break

if reset_word_btn:
    ta: TextAnalyzer = st.session_state["text_analyzer"]
    ta.word = []
    ta.reset_timer()


# ══════════════════════════════════════════════════════════════════════════════
#  DRAIN RESULT QUEUE & update session state
# ══════════════════════════════════════════════════════════════════════════════
ta: TextAnalyzer = st.session_state["text_analyzer"]
lc: str          = st.session_state["language_code"]

_drained = 0
while not result_q.empty() and _drained < 30:
    try:
        item = result_q.get_nowait()
    except queue.Empty:
        break
    _drained += 1

    pred       = item["prediction"]
    conf       = item["confidence"]
    st.session_state["confidence"] = conf

    if pred is not None:
        prev_len = len(ta.sentence)
        confirmed, _, _ = ta.process(pred)
        st.session_state["hold_progress"] = ta.hold_progress

        if confirmed is not None:
            st.session_state["last_sign"] = confirmed
            if confirmed not in ("SPACE", "BACKSPACE"):
                st.session_state["history"].append({"sign": confirmed, "confidence": conf})
                if len(st.session_state["history"]) > 100:
                    st.session_state["history"] = st.session_state["history"][-100:]

        if len(ta.sentence) > prev_len:
            if lc == "en":
                st.session_state["translated_str"] = " ".join(ta.sentence)
            else:
                st.session_state["translated_str"] = translate(" ".join(ta.sentence), lc)
    else:
        st.session_state["hold_progress"] = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SPEAK HANDLER
# ══════════════════════════════════════════════════════════════════════════════
if speak_btn:
    text_to_speak = st.session_state["translated_str"]
    if text_to_speak:
        with st.spinner("Generating audio…"):
            audio_bytes = _speak_audio(text_to_speak, lc)
        if audio_bytes:
            st.sidebar.audio(audio_bytes, format="audio/mp3", autoplay=True)
        else:
            st.sidebar.warning("Speech unavailable — check internet connection.")
    else:
        st.sidebar.info("Nothing to speak yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT  — video | metrics
# ══════════════════════════════════════════════════════════════════════════════
col_vid, col_panel = st.columns([1.5, 1], gap="medium")

with col_vid:
    webrtc_ctx = webrtc_streamer(
        key="isl-bridge",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=ISLVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_panel:

    # ── Current Sign ──────────────────────────────────────────────────────
    sign_now   = st.session_state["last_sign"] or "–"
    conf_val   = st.session_state["confidence"]
    hold_val   = st.session_state["hold_progress"]

    st.markdown(f"""
    <div class="metric-card">
      <div class="card-label">Current Sign</div>
      <div style="display:flex;align-items:flex-start;gap:1rem">
        <span class="sign-glyph">{sign_now}</span>
        <div style="flex:1;padding-top:0.3rem">
          {_progress_html(conf_val, "conf", f"CONF · {int(conf_val*100)}%")}
          {_progress_html(hold_val, "hold", f"HOLD · {int(hold_val*100)}%")}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Word in progress ──────────────────────────────────────────────────
    word_now = "".join(ta.word) or "–"
    st.markdown(f"""
    <div class="metric-card">
      <div class="card-label">Word in Progress</div>
      <span class="word-val">{word_now}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sentence ──────────────────────────────────────────────────────────
    sent_now = st.session_state["translated_str"] or ""
    st.markdown(f"""
    <div class="metric-card" style="min-height:80px">
      <div class="card-label">Sentence</div>
      <span class="sentence-val">{sent_now if sent_now else '<span style="color:#374151">— waiting for signs —</span>'}</span>
    </div>
    """, unsafe_allow_html=True)

    # Copy sentence button
    if sent_now:
        st.code(sent_now, language=None)

    # ── Sign history ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
      <div class="card-label">Sign History</div>
      {_chips_html(st.session_state["history"])}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-RERUN while WebRTC is active (keeps UI live)
# ══════════════════════════════════════════════════════════════════════════════
if webrtc_ctx.state.playing:
    import time
    time.sleep(0.12)      # ~8 fps UI refresh
    st.rerun()

