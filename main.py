from voice_modules import stt
from voice_modules import tts_offline as tts
url = "http://192.168.43.186:8080/shot.jpg"
from captioning_module import ic
# import objdetection as objdet


if __name__ == "__main__":
    i=0
    stt.calibrate()
    while True:
        command  = stt.stt(stt.mic_in())
        print("command: {}".format(command))
        if command == "hello" or command == "hi" or command == "what up" or command == "whats up" :
            tts.speak("Hello User, Good to see you!")
        elif command == "navigate" or command == "route" or command == "navigate environment":
            # objdet.navigate()
            pass
        elif command == "describe" or command == "caption" or command == "image captioning" or command == "describe environment" or command == "sense environment" or command == "sense":
            caption = ic.prediction_for(ic.fit(url=url))
            tts.speak("i sense that, "+caption)
            pass
        elif command == "exit" or command == "close":
            break
        else:
            print(command)
            tts.speak("unable to understand the voice command.")
            tts.speak("command detected was, {}".format(command))
        i+=1