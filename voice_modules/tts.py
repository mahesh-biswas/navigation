from gtts import gTTS
import playsound
import os
import time
"""
*THIS IS ONLINE MODULE*
package required:
        main{   
               gtts   :   pip install gTTS
        }
        side-packages{
                playsound             :   pip install playsound
        }
"""
def speak(text="stop!",lang='hi',slow=False,fname="speak.mp3"):
    ttsobj = gTTS(text=text, lang=lang, slow=False)
    ttsobj.save(fname)
    playsound.playsound(fname, True)
    os.remove(fname)

if __name__ == "__main__":
    speak("Hello this is a test track")
    speak("Move Left") 
    speak("Move Right") 
    speak("Stop") 
    speak("Stop, for god's sake") 
    pass