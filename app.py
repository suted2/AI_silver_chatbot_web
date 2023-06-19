# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import tempfile
import whisper
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import pyaudio 
import wave
import time
import uuid
import pandas as pd
import torch

import pymysql
from model_for_inference import Load_Model_Tokenizer
import os
import pickle
import torch
from 카테고리별inference import Category_Callcenter

# DB연동
db_conn = pymysql.connect(
    host= 'localhost',
    port= 3306,
    user= 'root',
    passwd= '1234',
    db= 'test',
    charset= 'utf8'
)

print(db_conn)
counts = dict()

# df_normal = pd.
# df_corona = 
# df_water = 

category_dict = {'일반행정' : 0,
              '코로나' : 1,
              '생활하수도' : 2,
              '대중교통' : 3}
category = -1

df0 = pd.read_pickle('normal_emb_id.pickle')
df1 = pd.read_pickle('corona_emb_id.pickle')
df2 = pd.read_pickle('water_emb_id.pickle')
df3 = pd.DataFrame()
dfs = [df0, df1, df2, df3]
print('embedding loaded!')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# emb_dir = 'datasets'
model_bert, tokenizer = Load_Model_Tokenizer('bert_model')   # model 저장 경로
model_bert.to(device)



model_STT = whisper.load_model("medium", device= device)

model_name = 'sgunderscore/hatescore-korean-hate-speech'
model_toxic = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipe = TextClassificationPipeline(
        model = model_toxic,
        tokenizer = tokenizer,
        # device = -1, # cpu: -1
        device = 0, # gpu: 0
        top_k = None,
        function_to_apply = 'sigmoid')

# Flask 객체 인스턴스 생성
app = Flask(__name__)

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

@app.route('/video_main') # 접속하는 url
def video_main():
    global category
    category = request.args.get('category')

    # polyy_infer(df[df['category'] == ]) 

    if 'id' in request.args:
        answer_id = request.args.get('id')

        count = counts[int(answer_id)] + 1
        cursor = db_conn.cursor()

        query = f'UPDATE faq SET count = {count} WHERE id = {answer_id}'

        cursor.execute(query)

        db_conn.commit()
        
        cursor = db_conn.cursor()

        query = f'select * from path_table where id = {answer_id}'

        cursor.execute(query)
        answer_id = [{'audio' : c[1], 'video' : c[2], 'answer' : c[3]} for c in cursor][0]

    else:
        answer_id = {'audio' : 'intro.wav', 'video' : 'intro.mp4', 'answer' : '정부에서 별도 지원금을 지급하고 있습니다.'}


    cursor = db_conn.cursor()
    query = f'select * from faq where category = {category_dict[category]} order by count desc limit 20'

    cursor.execute(query)
    
    result = []
    for idx, i in enumerate(cursor):
        counts[i[0]] = i[2]
        result.append({'id' : i[0], 'question' : i[1], 'count' : idx})

    return render_template('video_main.html', title = category, answer_id= answer_id, result= result)


@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = str(uuid.uuid4()) + ".wav"
    file_name = 'Temp' + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    # print(category)
    emb_df = dfs[category_dict[category]]
    # print(emb_df)

    text = infer_STT(full_file_name)
    if toxic_check(text):
        ment = "욕설이 감지되었습니다."
        # print(ment)
        return ment
    
    print(text)
    # for result in pipe(text)[0]:
        # print(result)
    answer_id = infer_POLY(text, emb_df)
    
    print(answer_id)

    cursor = db_conn.cursor()

    query = f'select * from path_table where id = {answer_id}'

    cursor.execute(query)
    answer_id = [{'audio' : c[1], 'video' : c[2], 'answer' : c[3]} for c in cursor][0]
    return answer_id
    
def infer_STT(filename):
    global model_STT
    result = model_STT.transcribe(filename, fp16=True)
    print()
    return result["text"]

def infer_POLY(text, emb_df):
    global model_bert
    global tokenizer
    call_center = Category_Callcenter(model=model_bert, tokenizer=tokenizer, emb_df=emb_df, device=device)
    emb_idx = call_center.inference(text)
    print(emb_idx)
    answer_id = emb_df.iloc[emb_idx]['index']
    return answer_id

def toxic_check(txt):
    result = pipe(txt)

    if result[0][0]['label'] != "None":
        return True
    
    else: 
        return False

if __name__ == '__main__':
    # app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    app.run(host='0.0.0.0', port='8080', debug=True, threaded=True)