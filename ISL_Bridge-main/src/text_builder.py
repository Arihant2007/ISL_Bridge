import time
from groq import Groq
from dotenv import load_dotenv
import os

CONFIRM_THRESHOLD = 1.5
load_dotenv()


class TextAnalyzer:
    def __init__(self):
        self.sign = None
        self.word = []
        self.sentence = []
        self.start = None
        self.hold_progress: float = 0.0
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

    def process(self, raw_sign):
        confirmed_sign = self._confirm_sign(raw_sign)
        if confirmed_sign == "BACKSPACE":
            self._handle_backspace()
        elif confirmed_sign == "SPACE":
            self.sentence = self._build_sentence()
        elif confirmed_sign is not None:
            self.word = self._add_letter(confirmed_sign)
        return confirmed_sign, self.word, self.sentence

    def reset_timer(self):
        self.start = None
        self.hold_progress = 0.0

    def _confirm_sign(self, raw_sign):
        now = time.time()
        if raw_sign != self.sign:
            self.start = now
            self.sign = raw_sign
            self.hold_progress = 0.0
        else:
            elapsed = now - self.start if self.start else 0
            self.hold_progress = min(elapsed / CONFIRM_THRESHOLD, 1.0)
            if elapsed >= CONFIRM_THRESHOLD:
                self.start = now
                self.hold_progress = 1.0
                return self.sign
        return None

    def _add_letter(self, confirmed_sign):
        self.word.append(confirmed_sign)
        return self.word

    def _handle_backspace(self):
        if self.word:
            self.word.pop()

    def _build_sentence(self):
        if not self.word:
            return self.sentence
        letters = "".join(self.word)

        # Without a configured key, keep the spelled word intact as a safe fallback.
        if self.client is None:
            word = letters
        else:
            prompt = (
                f"Interpret this noisy ISL fingerspelled sequence as one intended English word: {letters}. "
                f"Return only one uppercase word of approximately {len(letters)} letters."
            )
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[{"role": "user", "content": prompt}],
                )
                word = (response.choices[0].message.content or "").strip().split()[0].upper()
            except Exception:
                word = letters
            if not word:
                word = letters

        self.sentence.append(word)
        self.word = []
        return self.sentence
