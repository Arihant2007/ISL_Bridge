import time
from groq import Groq
from dotenv import load_dotenv
import os

# How long (seconds) a sign must be held to be confirmed
CONFIRM_THRESHOLD = 1.5

load_dotenv()


class TextAnalyzer:

    def __init__(self):
        self.sign = None
        self.word = []
        self.sentence = []
        self.start = None
        # Progress of current sign hold, 0.0 – 1.0
        self.hold_progress: float = 0.0
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw_sign):
        """Process one frame's raw prediction.

        Returns (confirmed_sign | None, word_letters, sentence_words).
        """
        confirmed_sign = self._confirm_sign(raw_sign)

        if confirmed_sign is None:
            pass
        elif confirmed_sign == "BACKSPACE":
            self._handle_backspace()
        elif confirmed_sign == "SPACE":
            self.sentence = self._build_sentence()
        else:
            self.word = self._add_letter(confirmed_sign)

        return confirmed_sign, self.word, self.sentence

    def reset_timer(self):
        """Call this when the session is paused so the hold-timer restarts on resume."""
        self.start = None
        self.hold_progress = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confirm_sign(self, raw_sign):
        now = time.time()

        if raw_sign != self.sign:
            # New sign detected — restart timer
            self.start = now
            self.sign = raw_sign
            self.hold_progress = 0.0
        else:
            elapsed = now - self.start if self.start else 0
            self.hold_progress = min(elapsed / CONFIRM_THRESHOLD, 1.0)
            if elapsed >= CONFIRM_THRESHOLD:
                self.start = now  # reset so it doesn't fire every frame
                self.hold_progress = 1.0
                return self.sign

        return None

    def _add_letter(self, confirmed_sign):
        self.word.append(confirmed_sign)
        return self.word

    def _handle_backspace(self):
        """Remove the last letter from the current word."""
        if self.word:
            self.word.pop()

    def _build_sentence(self):
        if not self.word:
            return self.sentence

        letters = "".join(self.word)  # string

        prompt = (
            f"The user has gestured a sequence of letters from Indian Sign Language along with "
            f"numbers from 0 to 9, and you are receiving this sequence as {letters}. These letters, read in "
            f"the SAME ORDER they are given, are meant to spell a single word, but the spelling is not perfect, "
            f"and it is YOUR JOB to figure out what that word is. The sequence may contain EXTRA letters that don't "
            f"belong, it may be MISSING some letters, or it may include a few WRONG letters, so you must look past these "
            f"mistakes and work out the correct intended word. The wrong letters will at most times be in the MIDDLE of the letters. "
            f"First try removing duplicate consecutive letters and see if a word forms, before trying to replace a letter. "
            f"For sequences of 1-2 letters that form common English words like I, A, AM, IS, IT, BE, DO — return them exactly as they are. "
            f"The word you return must be a SINGLE word and never a phrase, "
            f"it must be written entirely in UPPERCASE, and its length should be SIMILAR to the number of letters given — for "
            f"example, if only 2 letters are provided, the word should be roughly 2 to 4 letters long rather than something much "
            f"longer or shorter — and as a guide, the word has APPROXIMATELY {len(self.word)} letters. Do NOT explain your answer, "
            f"do NOT ask any questions, and do NOT add anything else whatsoever: respond with ONLY the single uppercase word and "
            f"nothing more, because returning anything other than a single uppercase word means you have failed at this task."
        )

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )

        word = response.choices[0].message.content.strip()

        if not word:
            word = letters

        self.sentence.append(word)
        self.word = []

        return self.sentence

