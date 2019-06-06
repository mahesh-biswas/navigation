import speech_recognition as sr
import time
"""
    package required:
        main{
               speech_recognition   :   pip install SpeechRecognition
        }
        side-packages{
                pyaudio             :   pip install pyaudio
                pocketsphinx        :   pip install pocketsphinx 
        }
"""
r = sr.Recognizer()

def calibrate():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
    print("calibration complete..")


def mic_in(timeout=3, phrase_limit=4):
    """
        takes input from MICROPHONE and returns the audio object.
    """
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        print("Processing...")
    return audio

def stt(audio_source):
    """
        takes audio object as input and returns the generated TEXT.
        if internet is available then gstt module is used(very high accuracy).
        else offline sphinx module is used(low accuracy).
    """
    try:
        # return "Google: {}".format(stt_google(audio_source))
        return stt_google(audio_source)
    except sr.UnknownValueError:
        print("could not understand audio")
    except sr.RequestError as e:
        print("Network Error: {}".format(e))
        return "Offline: {}".format(stt_offline(audio_source))

def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def stt_google(audio_source):
    """
        calls the gstt module
    """
    return r.recognize_google(audio_source)

def stt_offline(audio_source):
    """
        calls the sphinx module
    """
    return r.recognize_sphinx(audio_source)

if __name__ == "__main__":
    # calibrate()
    print(stt(mic_in()))
    