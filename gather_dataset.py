import os
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

for folder in range(2):
	folder_path = 'data/{}'.format(folder)

	if os.path.exists(folder_path) == False:
		os.mkdir(folder_path)

	for idx in range(10):

		print('talk now...')
		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
		sd.wait()  # Wait until recording is finished

		filename = 'data/{}/sample_{}_{}.wav'.format(folder, folder, idx)
		write(filename, fs, myrecording)  # Save as WAV file 
		print('saved {}\n'.format(filename))
		