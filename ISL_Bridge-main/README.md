---
title: ISL Bridge
emoji: 🤝
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# ISL Bridge
### Real-time Indian Sign Language Fingerspelling Recognition System

ISL Bridge is a real-time tool that helps hearing people understand Indian Sign Language. It detects fingerspelled letters via webcam or a video file, forms words using an LLM, and translates them into any language with optional speech output.


# Demo
**Live Demo:** [ISL Bridge on Hugging Face Spaces](https://huggingface.co/spaces/ShridaliSingh/ISL_Bridge)

**Docker Image:** [ISL Bridge on Docker Hub](https://hub.docker.com/r/shridalisingh/isl-bridge)


# Features 
- Real-time fingerspelling detection via webcam or video file input.
- Primary user is hearing person trying to understand a deaf signer.
- Supports 34 signs - 23 ISL letters (excluding dynamic signs H,J,Y) and 10 ISL digits (0-9) and SPACE gesture.
- SPACE gesture (dominant hand fist) = word boundary signal.
- Left-handed signing supported via data augmentation.
- Brief accidental detections automatically ignored - signs must be held for 1.5 seconds to confirm.
- LLM (Groq) handles noisy/imperfect letter sequences for word formation.
- Multilingual translation - user types language name at startup.
- Unicode text rendering on screen - supports Hindi, Punjabi, Arabic, Tamil, Telugu, Bengali and all Latin scripts.
- Optional speech output in target language via gTTS - press 'S' to speak.


# How it works
## ML Pipeline
- Collected own data since no reliable datasets avaibale for ISL static signs.
- Opened Camera thru OpenCV, collected hand landmarks using MediaPipe and saved it in a csv file.
- The csv contained 42 hand landmarks (21 for each hand), each handmark comprised of x - y - z coordinated ( 21 x 2 x 3 = 126 coordinates) and a sign label in each row.
- Total 5+ data collection sessions with slightly different lighting and angles to ensure robustness.
- Around 12-14 thousand entries in the csv.
- Model comprised of 5 layers and 34 classes.
- Input layer - 126 neurons, 1st hidden layer - 256 neurons, 2nd hidden layer - 128 neurons, 3rd hidden layer - 64 neurons and output layer - 34 neurons.
- Achieved as accuracy of **97.71%** 
- Dropout regularization to prevent overfitting
- Class weighting to handle imbalanced dataset.
- Data augmentation - Coordinate flipping and hand swapping to support left-handed signers.
- Tracked the test accuracy while training to keep check on overfitting.
- Saved the best stage of the model as the final model. 

## App Pipeline
- OpenCV provided frames one at a time.
- MediaPipe read the hand landmarks.
- The 126 coordinates passed it into the model.
- Model returned a tensor of probabilites for each class.
- Argmax used to predict the sign.
- Predicted sign confirmed only if held for 1.5s.
- Confirmed sign passed to list of letters in a word.
- SPACE gesture marked the end of a word.
- The letters list passed to the sentence formation logic, where the llm guesses the word.
- The words concatenated together to form a sentence.
- The sentence passed to GoogleTranslator for translation into the user selected language and also passed to the gTTS for speech functionality.
- Translation triggered when a new word is added.
- The sign, word and sentence displayed on the OpenCV screen with the help of Pillow.
- Noto Sans fonts used with Pillow to render Unicode text on screen. 
- Press 'S' for speech feature and 'Esc' to terminate the program.


# Startup/Installation 
- Clone the repo.
- Open terminal in the project folder.
- Create a virtual environment with Python 3.11 and activate it.(Windows: .venv\Scripts\activate, Mac/Linux: source .venv/bin/activate)
- Note: Python 3.11 required - MediaPipe's solutions API is incompatible with newer versions.
- Add Groq API key as `GROQ_API_KEY=your_key_here` in a file called .env in the project root. The API key can be fetched from console.groq.com.
- Run `pip install -r requirements.txt` to install all the dependencies.
- Note: `requirements.txt` installs CPU-only PyTorch, sufficient for running the app. 
- For retraining the model with GPU support, install torch separately: `pip install torch==2.12.0`
- The trained model is included in the repo. If the user wants to retrain with their own data, run `collect_data.py` first, then `clean_data.py`, then `train_model.py`.
- Note: `collect_data.py` is a personal data collection script with minimal input validation - designed for developer use, not end-user use.


# Usage 
## Data collection
- Run `collect_data.py`.
- Select camera input or video file input, then select the sign that you want to record. (The sign cannot be dynamic - H,J,Y).
- Note: The video file should contain one sign per file as for one file, one input for the label will be asked.
- After recording a sign, press 'enter' to stop recording and input the next sign. Stop the particular recording session by pressing 'Esc'.
- Run `clean_data.py` to clean the data.
- Run `delete_data.py` and enter the signs to delete them from your dataset.

## Application
- Run `python app.py` to start the application.
- Select camera input or video file input, enter the language you want the text/audio to be in, and answer the prompt for speech functionality.
- Record a sign by keeping it steady for 1.5s.
- After fingerspellign a word, show SPACE gesture (dominant hand fist) to confirm a word boundary. 
- Press 'S' if you want to listen to the sentence formed till that point (will work only if you entered y for the prompt).
- Press 'Esc' to stop recording.

## Docker 
- Pull the image: `docker pull shridalisingh/isl-bridge`
- Run the container: `docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here shridalisingh/isl-bridge`
- Replace `your_key_here` with your own Groq API key from console.groq.com.
- Open your browser and go to `localhost:7860`.

## Flask Web Application
- Run `python flask_app/app.py` from the project root.
- Open your browser and go to `localhost:7860`.
- Select camera or video file input.
- Select your language from the dropdown.
- Click Start and begin signing.
- Use the Speak button for audio output.
- Use Reset to clear and start over.
- Use End to stop the session.


# Known Limitations
- This is a fingerspelling system, not full ISL word or phase translation.
- `Collect_data.py` minimal validation - designed for developer use, not robust enough for general users to collect their own data without understanding the system.
- Left-handed support via augmentation - works but slightly less accurate than right-handed signing since actual left-hand data wasn't recorded.
- H, J, Y excluded - dynamic signs requiring movement.
- Dataset is small and personal - recorded by one person across 5+ sessions.
- W detection unreliable - known MediaPipe limitation.
- Lighting sensitivity - MediaPipe landmark accuracy drops in poor lighting, affecting predictions.
- LLM word formation imperfect - can struggle with uncommon words or heavily corrupted letter sequences.
- Requires internet - Groq and gTTS both need it.
- gTTS uses Google Translate's public endpoint - free, no API key needed.
- deep_translator used for translation - free, no API key needed.
- Fonts folder included only 6 Noto Sans font files for Unicode rendering.
- Video pause/resume timer issue - TextAnalyzer timer continues running during pause, sign held at pause time may confirm immediately on resume.
- Deployed version on Hugging Face free tier may feel slower than local - frames processed on shared CPU, causing higher inference latency.


# Tech Stack 
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| NumPy | Math and arrays - KNN and neural network from scratch |
| OpenCV | Webcam/video capture and frame processing |
| MediaPipe | Hand landmark extraction - 21 points per hand |
| scikit-learn | Train/test split |
| PyTorch | Production classifier training and inference |
| Groq API | LLM for intelligent word formation from fingerspelled letters |
| python-dotenv | Load API key from .env file |
| gTTS | Text to speech in target language |
| pygame | Audio playback |
| deep-translator | Multilingual translation via Google Translate |
| Pillow | Unicode text rendering on OpenCV frames |
| threading | Non-blocking speech playback |
| Flask | Web application and deployment |
| Docker | Containerisation for portable deployment |

# Future Scope
- Expansion of dataset with more signers.
- LLM prompt improvement.
- Client-side inference via ONNX to eliminate server-side latency in the deployed version.
- LSTM for dynamic signs (H,J,Y) - stretch goal, not a confirmed next step.







