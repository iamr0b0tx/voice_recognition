# from standard library
import os, json, random, glob, subprocess

# from third party lib
from flask import Flask, render_template, request

# from lib code

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
PORT = 3000
DEBUG_STATE = False
SESSION = False

@app.route('/')
def index():
	global SESSION;
	if SESSION == False:
		return render_template('index.html')
	
	else:
		return render_template('site.html')

@app.route('/listen')
def listen():
	global SESSION;
	status, text = True, 0
	try:
		test = subprocess.Popen(["python", "test.py"], stdout=subprocess.PIPE)
		text = int(test.communicate()[0].decode())

	except Exception as e:
		status, text = False, 0

	print(text)
	SESSION = bool(text)

	return json.dumps({'status':status, 'data':text})

if __name__ == '__main__':
	port = int(os.environ.get('PORT', PORT))
	app.run(host='0.0.0.0', port=port, debug=DEBUG_STATE)
