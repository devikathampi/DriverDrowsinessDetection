import time
import speech_recognition as sr
from gtts import gTTS
import os
from pygame import mixer  
import random

def recognize_speech_from_mic(recognizer, microphone):
    
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

if __name__ == "__main__":
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    instructions = ("Repeat the words after me\n")
    print(instructions)
    myobj = gTTS(text=instructions, lang='en', tld='co.uk', slow=False)
    myobj.save("STT.mp3")
    mixer.init()
    mixer.music.load('STT.mp3')
    mixer.music.play()
    time.sleep(2)
    #WORDS = ["Banana", "Orange", "Mango", "Apple", "Lemon","Grape","Litchi", "Cherry", "Peach", "Papaya"]
    WORDS = ["acknowledgement", "happiness", "formula", "elegance", "tripod", "washington", "football",
             "spectacles", "christmas", "adversity", "applause", "controversy", "framework",
            "fireworks", "harassment", "heritage", "innovation", "livelihood", "pandemic", "phenomenon","condescending", 
            "rehabilitation", "sustainable", "transformation", "xenophobia", "australia","america",
            "environment","administration","animation","agreement","community","development"]
    c = 0
    DrowsyNumber = 3 #number of incorrect words for getting drowsy message
    WordsAsked = 7 #number of words to be asked
    
    for i in range(WordsAsked):
        #word = WORDS[i]
        word = random.choice(WORDS)
        for j in range(5):
            x=('Say {n}').format(n=word)
            print(x)
            myobj = gTTS(text=x, lang='en', tld='co.uk', slow=False)
            file1 = str("hello" + str(i) + ".mp3")
            myobj.save(file1)
            mixer.init()
            mixer.music.load(file1)
            mixer.music.play()
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("I didn't catch that. What did you say?\n")
            i=i+1
        if guess["error"]:
            print("ERROR: {}".format(guess["error"]))
            break

        print("You said: {}".format(guess["transcription"]))

        guess_is_correct = guess["transcription"].lower() == word.lower()
        

        if guess_is_correct:
            myobj = gTTS(text="Correct!", lang='en', tld='co.uk', slow=False)
            myobj.save("STT.mp3")
            mixer.init()
            mixer.music.load('STT.mp3')
            mixer.music.play()
            print("Correct!\n".format(word))
            time.sleep(1)
        else:
            myobj = gTTS(text="Incorrect!", lang='en', tld='co.uk', slow=False)
            myobj.save("STT.mp3")
            mixer.init()
            mixer.music.load('STT.mp3')
            mixer.music.play()
            print("Incorrect!\n")
            time.sleep(1)
            c = c+1
        os.remove(file1)
    if(c >= DrowsyNumber):
        #myobj = gTTS(text="You are drowsy! Please stop driving immediately and seek assistance", lang='en', tld='co.uk', slow=False)
        #myobj.save("STT.mp3")
        mixer.init()
        mixer.music.load('soundfiles/Drowsy.mp3')
        mixer.music.play()
        print("Drowsy")
        time.sleep(7)
    else:
        #myobj = gTTS(text="Looks like you are alert! You are good to go!", lang='en', tld='co.uk', slow=False)
        #myobj.save("STT.mp3")
        mixer.init()
        mixer.music.load('soundfiles/Alert.mp3')
        mixer.music.play()
        print("Not Drowsy")
        time.sleep(7)
        
    print("TMM done")