# ISL Bridge

Real-time Indian Sign Language fingerspelling demo built with MediaPipe, PyTorch, Streamlit WebRTC, optional Groq word cleanup, translation, and speech output.

## Run locally

Use Python 3.11. From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run ISL_Bridge-main/streamlit_app.py
```

The app works without a Groq key; it will use the spelled letters as the word. Set `GROQ_API_KEY` in Streamlit secrets to enable optional word cleanup.

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and connect GitHub.
2. Create an app from `Arihant2007/ISL_Bridge`, branch `main`.
3. Set the app file to `ISL_Bridge-main/streamlit_app.py`.
4. Deploy. Add `GROQ_API_KEY` under the app's **Settings → Secrets** only if you want LLM spelling cleanup.

The model and label checkpoint files in the repository are Git LFS pointer files, not the actual checkpoints. The app now starts safely and shows this limitation, but gesture predictions require replacing those pointers with the real files (or configuring Git LFS download for the deployment).

## Project scope

This is fingerspelling recognition for a limited static-sign vocabulary. It does not translate full ISL grammar or dynamic signs H, J, and Y. Camera use requires HTTPS and browser permission.
