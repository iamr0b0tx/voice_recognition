# from standard library
import os, json, random, glob

import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished

# from third party lib
from flask import Flask, render_template, request
from train import model, audio2vector

# from lib code
model.layers[2].load_weights('weights/weights.h5')

anchor_audios = glob.glob('data/{}/*.wav'.format(sorted(os.listdir('data'))[0]))
anchor_audio = audio2vector(anchor_audios[random.randint(0, len(anchor_audios) - 1)])

threshold = 0.3
with open('weights/threshold.txt') as f:
	threshold = float(f.read())

text = 0

try:
	write('sample.wav', fs, myrecording)  # Save as WAV file 

	audio = audio2vector('sample.wav')
	prediction = model.predict([[audio.reshape((20, 400, 1))], [anchor_audio.reshape((20, 400, 1))]])
	prediction = prediction[0, 0]

	text = 1 if prediction < threshold else 0
	# print(threshold, prediction)

except Exception as e:
	pass

print(text, end='')


