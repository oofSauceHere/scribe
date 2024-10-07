import os
import time
import speech_recognition as sr
# import train
import hwt

# process text to ensure well-formed sentences
def process(res, index):
    out = open("mytext.txt", "w")
    res = res[0:index].strip()
    out.write(res)
    out.close()

# callback function that speech_recognition uses to process audio as it's heard
def callback(recognizer, audio):
    try:
        words = recognizer.recognize_whisper(audio, language="en")
        out = open("mytext.txt", "a")
        out.write(words)
        out.close()
        # if "print" in words.lower():
        #     exit(1)
        # time.sleep(1)
    except sr.UnknownValueError:
        print("couldn't understand")
    except sr.RequestError as e:
        print(f"couldn't request, {e}")
    except:
        exit(1)

# code that handles background listening for text (in its own thread so speech can be continually processed until
# the word "print" is heard)
def listening_thread():
    # all of this code is taken directly from the Nathan Bot b/c speech recognition requires the same process

    # sets up speech recognizer and microphone input
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=sr.Microphone.list_microphone_names().index("Voicemeeter Out B2 (VB-Audio Vo"))

    # MIC DEBUGGING

    # for index, name in enumerate(sr.Microphone.list_microphone_names()):
    #     print(name)

    # tailors model to ignore ambient noise
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

        # for one-time transcription
        # print("talk")
        # audio = recognizer.record(source, duration=30)
        # print("done")
        # words = recognizer.recognize_whisper(audio, language="en").lower()
        # out = open("mytext.txt", "w")
        # out.write(words)
        # out.close()
    
    # clears mytext file
    out = open("mytext.txt", "w")
    out.close()

    # runs listener in a separate thread
    stop_listening = recognizer.listen_in_background(mic, callback)
    print("running !!")

    # keep going forever!!
    while True:
        time.sleep(0.5) # don't want to process too frequently (it's unnecessary)
        read = open("mytext.txt", "r")
        res = read.read()
        read.close()

        # if we hear "print" then we exit the loop and dump to mytext
        index = res.lower().find("print")
        if not index == -1:
            stop_listening()
            process(res, index) # necessary formatting step
            print("done !!")
            return

def main():
    # listening_thread()
    hwt.main()

if __name__ == "__main__":
    main()