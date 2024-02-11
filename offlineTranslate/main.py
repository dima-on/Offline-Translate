from vosk import Model, KaldiRecognizer
import time, json, pyaudio
import os
import torch
import sounddevice as sd
from transformers import MarianMTModel, MarianTokenizer
from typing import Sequence
import threading

typeLeg = 0 #0 - En, 1 - Ru Read

textDialog = []
numDialog = 0
numSpeak = 0
InDialog = 0

modelRu = Model("vosk-model-small-ru-0.4")
modelEn = Model("vosk-model-small-en-us-0.15")
recRu = KaldiRecognizer(modelRu, 16000)
recEn = KaldiRecognizer(modelEn, 16000)
micro = pyaudio.PyAudio()
MicroStream = micro.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)




device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'
local_fileEn = 'modelEn.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                   local_file)
if not os.path.isfile(local_fileEn):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/en/v3_en.pt',
                                   local_fileEn)


modelVoice = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
modelVoiceEn = torch.package.PackageImporter(local_fileEn).load_pickle("tts_models", "model")
modelVoice.to(device)
modelVoiceEn.to(device)


sample_rate = 48000
speaker='baya' # aidar, baya,
speakerEn='en_0' # aidar, baya,

def StopSpeak():
    global numSpeak
    global InDialog
    numSpeak +=1
    InDialog = 0
    #sd.stop()
    print("daw")

def va_speak(what: str):
    InDialog = 1
    try:
        print(what)
        audio = modelVoice.apply_tts(text=what+"..",
                                speaker=speaker,
                                sample_rate=sample_rate)

        sd.play(audio, sample_rate * 1.05)
        timerForStop = threading.Timer((len(audio) / sample_rate) + 0.5, StopSpeak())

    except:
        print("DontWork")
        StopSpeak()

def va_speakEn(what: str):
    InDialog = 1

    try:
        audio = modelVoiceEn.apply_tts(text=what+"..",
                                speaker=speakerEn,
                                sample_rate=sample_rate)

        sd.play(audio, sample_rate * 1.05)
        timerForStop = threading.Timer((len(audio) / sample_rate) + 0.5, StopSpeak())

    except:
        print("DontWork")
        StopSpeak()


MicroStream.start_stream()

def listenRu():
    while True:
        if InDialog == 0:
            if len(textDialog) - 1 >= numSpeak:
                va_speak(textDialog[numSpeak])
        data = MicroStream.read(4000, exception_on_overflow=False)
        if recRu.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(recRu.Result())
            if answer['text']:
                yield answer['text']

def listenEn():
    while True:
        if InDialog == 0:
            if len(textDialog) - 1 >= numSpeak:
                va_speak(textDialog[numSpeak])
        data = MicroStream.read(4000, exception_on_overflow=False)
        if recEn.AcceptWaveform(data) and len(data) > 0:
            answer = json.loads(recEn.Result())
            if answer['text']:
                yield answer['text']

class Translator:
    def __init__(self, source_lang: str, dest_lang: str) -> None:
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        tokens = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        translate_tokens = self.model.generate(**tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translate_tokens]


marian_ru_en = Translator('ru', 'en')
marian_en_ru = Translator('en', 'ru')

if typeLeg == 0:
    for text in listenEn():

        ruText = marian_en_ru.translate([text])
        ruText = str(ruText)
        print(text + " - " + ruText)
        if ruText != text:
            textDialog.append(1)
            textDialog[numDialog] = ruText
            numDialog += 1
else:

    for text in listenRu():

        ti = time.time()
        enText = marian_ru_en.translate([text])
        enText = str(enText)
        if enText != text:
            textDialog.append(1)
            textDialog[numDialog] = enText
            numDialog += 1




