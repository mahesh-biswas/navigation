
import pyttsx3

"""
*THIS IS OFFLINE MODULE*
package required:
        main{
               pyttsx3   :   pip install pyttsx3
        }
"""
engine = pyttsx3.init()
engine.setProperty('rate', 165)
def speak(text):
        engine.say(text)
        engine.runAndWait() 

def test_voice_speed(text):
        x = engine.getProperty('rate')
        for i in range(x,0,-5):
                print(i)
                engine.setProperty('rate', i)
                speak(text)


def test_voice_speak(text="This is a test message."):
        voices = engine.getProperty('voices')
        for voice in voices:
                engine.setProperty('voice', voice.id)
                print(voice.id)
                engine.say(text)
        engine.runAndWait()

if __name__ == "__main__":
        speak("Hello this is a test track")
        speak("Move Left") 
        speak("Move Right") 
        speak("Stop") 
        speak("Stop, for god's sake") 
        # engine.say("Move Right") 
        # engine.say("Stop") 
        # engine.say("Stop, for god's sake") 
        engine.runAndWait() 
