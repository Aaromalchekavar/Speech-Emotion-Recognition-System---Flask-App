#flask imports
from flask import Flask, render_template, request ,jsonify
import os
import json
from emotion import *

#install ffmpeg if error


#flask code

app = Flask(__name__,template_folder='./templates')


# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save():
    # Save the audio file as WAV
    audio_file = request.data
    file_path = os.path.join('/home/aaromalchekavar/Downloads/chatbot-emotion', 'audio.wav')
    with open(file_path, 'wb') as f:
        f.write(audio_file)
    return jsonify({"audio_file_path": file_path})

@app.route('/predict', methods=['POST'])
def predict():
    # Call predictEmotion method with saved file path
    file_path = request.get_json()['audio_file_path']
    path = json.loads(file_path)
    emotion = realtimepredict(path['audio_file_path'])
    return jsonify({"emotion": emotion})



if __name__ == '__main__':
    app.run()