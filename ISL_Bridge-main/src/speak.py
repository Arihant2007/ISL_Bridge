from gtts import gTTS
import pygame
import io
import threading

pygame.mixer.init()

def speak(sentence,language):
   
    audio = gTTS(text = sentence, lang = language)
    loc = io.BytesIO()
    audio.write_to_fp(loc)

    loc.seek(0)
    if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    

    def target():
        pygame.mixer.music.load(loc)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() :
            pass
    
    t = threading.Thread(target=target)
    t.start()