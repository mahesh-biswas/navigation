from captioning_module import ic
from voice_modules import tts_offline as tts
import time

url = "http://192.168.43.48:8080/shot.jpg"
def sleep(delay=1):
    time.sleep(delay)

while True:
    caption = ic.prediction_for(ic.fit(url=url))
    print("i sense that, "+caption)
    tts.speak("i sense that, "+caption)
    sleep(3)