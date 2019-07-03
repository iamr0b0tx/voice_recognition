# from standard library
import os, json, random, glob
import subprocess

# from third party lib
from flask import Flask, render_template, request
from train import model, audio2vector

import sounddevice as sd
from scipy.io.wavfile import write

# from lib code
model.layers[2].load_weights('weights/weights.h5')

anchor_audios = glob.glob('data/{}/*.wav'.format(sorted(os.listdir('data'))[0]))
anchor_audio = audio2vector(anchor_audios[random.randint(0, len(anchor_audios) - 1)])

threshold = 0.3
with open('weights/threshold.txt') as f:
	threshold = float(f.read())

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
PORT = 3000
DEBUG_STATE = False

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/listen')
def listen():
	status = True

	test = subprocess.Popen(["python", "test.py"], stdout=subprocess.PIPE)
	text = test.communicate()[0]
	print(text)

	return json.dumps({'status':status, 'data':text})

if __name__ == '__main__':
	port = int(os.environ.get('PORT', PORT))
	app.run(host='0.0.0.0', port=port, debug=DEBUG_STATE)
