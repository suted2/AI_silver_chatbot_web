# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import speech_recognition as sr
import os
import tempfile
import whisper
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import pyaudio 
import wave
import time
import uuid
import os 

Title_dict = {'admin' : '일반 행정',
            'corona' : '코로나',
            'sewage' : '생활하수도',
            'transport' : '대중 교통'}

model = whisper.load_model("medium")
model_name = 'sgunderscore/hatescore-korean-hate-speech'
model_toxic = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


pipe = TextClassificationPipeline(
        model = model_toxic,
        tokenizer = tokenizer,
        device = -1, # gpu: 0
        top_k = None,
        function_to_apply = 'sigmoid')

# Flask 객체 인스턴스 생성
app = Flask(__name__)
app.secret_key = os.urandom(24)


app.config["UPLOAD_DIR"] = "uploads"
app.config["UPLOAD_FOLDER"] = "audio"

recording = False
temp_file = "temp.wav"  # 임시 파일 경로

@app.route('/') # 접속하는 url
def main():
    return render_template('main.html')

@app.route('/main') # 접속하는 url
def to_main():
    return render_template('main.html')


category = ''

@app.route('/video_main') # 접속하는 url
def video_main():
    global category
    category = request.args.get('id')
    
    return render_template('video_main.html', title = Title_dict[category])


@app.route('/save-record', methods=['GET', 'POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)


    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        #flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".wav"
    file_name = 'Temp' + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    text = infer_STT(full_file_name)
    if toxic_check(text):
        ment = "욕설이 감지되었습니다."
        print(ment)
        return ment 
    else:

        ment = "정상적인 음성입니다."
        print(ment)
        return ment
    #return "<h2>success</h2>"




def infer_STT(filename):
    global model
    result = model.transcribe(filename, fp16=False)
    
    return result["text"]


def toxic_check(txt):
    result = pipe(txt)

    if result[0][0]['label'] != "None":
        return True
    
    else: 
        return False


if __name__ == '__main__':
    # app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    app.run(host='0.0.0.0', port='4000', debug=True, threaded=True)