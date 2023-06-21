# app.py
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import whisper
from transformers import TextClassificationPipeline, BertForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
import numpy as np

import pymysql
from model_for_inference import Load_Model_Tokenizer
import os
import torch
from inference import Callcenter

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

# 필요 함수 및 변수 정의
category_dict = {'일반행정' : 0,
                '코로나' : 1,
                '생활하수도' : 2,
                '대중교통' : 3}

person_dict = {'0' : '지창욱',
                '1' : '유재석'}
category = -1
person_id = 0

df0 = pd.read_pickle('dbPickle/normal_emb_id.pickle')
df1 = pd.read_pickle('dbPickle/corona_emb_id.pickle')
df2 = pd.read_pickle('dbPickle/water_emb_id.pickle')
df3 = pd.DataFrame()
dfs = [df0, df1, df2, df3]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("연결 Device :", device)

poly_dir = 'models/poly_encoder'
cross_dir = 'models/cross_encoder'
cross_encoder, _ = Load_Model_Tokenizer(cross_dir, model_type='cross')
poly_encoder, tokenizer_poly = Load_Model_Tokenizer(poly_dir, model_type='poly')

cross_encoder.to(device)
poly_encoder.to(device)

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

def infer_STT(filename):
    global model_STT
    result = model_STT.transcribe(filename, fp16=True)
    print()
    return result["text"]

def infer_POLY(text, emb_df):
    global tokenizer_poly
    call_center = Callcenter(poly_encoder=poly_encoder, cross_encoder=cross_encoder,
                        tokenizer=tokenizer_poly, emb_df=emb_df, device=device, topk= 10)
    top_k_cross_scores, top_k_indices = call_center.inference(text)
    print(top_k_cross_scores)
    print(top_k_indices)
    return top_k_cross_scores, top_k_indices

def toxic_check(txt):
    result = pipe(txt)
    if result[0][0]['label'] != "None":
        return True
    else: 
        return False

# Flask 객체 인스턴스 생성
app = Flask(__name__)

app.config["UPLOAD_DIR"] = "uploads"
app.config["UPLOAD_FOLDER"] = "audio"

recording = False
temp_file = "temp.wav"  # 임시 파일 경로

sub_stat = True

@app.route('/') # 상담원을 선택할 수 있는 첫 페이지
def main():
    return render_template('first.html')

@app.route('/main') # 카테고리 선택할 수 있는 페이지
def to_main():
    global person_id
    person_id = request.args.get('person_id')
    person = person_dict[person_id]
    return render_template('main.html', person_id=person_id, person=person)

@app.route('/video_main') # 상담을 진행하는 페이지
def video_main():
    global person_id
    global category
    category = request.args.get('category')

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
        answer_id = [{'audio' : person_id + '_' + c[1], 'video' : person_id + '_' + c[2], 'answer' : c[3]} for c in cursor][0]
    else:
        answer_id = {'audio' : f'{person_id}_intro2.wav', 'video' : f'{person_id}_intro2.mp4', 'answer' : '하단 중앙의 마이크 버튼을 눌러 시작, 종료하셔서 상담을 진행하실 수 있습니다.'} # 인트로용 멘트로 변경

    # DB에서 필요 데이터 불러오기
    cursor = db_conn.cursor()
    query = f'select * from faq where category = {str(category_dict[category])} order by count desc limit 20'
    cursor.execute(query)
    
    result = []
    for idx, i in enumerate(cursor):
        counts[i[0]] = i[2]
        result.append({'id' : i[0], 'question' : i[1], 'count' : idx})

    return render_template('video_main.html', title = category, answer_id= answer_id, result= result)


@app.route('/video_main', methods=['POST']) # 텍스트로 질문하기
def txt_input():
    global person_id
    global category
    category = request.args.get('category')
    print(category)
    
    text = request.form['question']
    
    if toxic_check(text): # 욕설 감지
        answer = "욕설이 감지되었습니다."
        return {'answer' : answer, 'text' : text}
    
    print(text)

    emb_df = dfs[category_dict[category]]
    top_k_cross_scores, top_k_indices = infer_POLY(text, emb_df)
    max_idx = np.argmax(top_k_cross_scores)

    if top_k_cross_scores[max_idx] > 10:
        emb_idx = top_k_indices[max_idx]
        answer_id = emb_df.iloc[emb_idx]['index']
        print(answer_id)

        cursor = db_conn.cursor()

        query = f'select * from path_table where id = {answer_id}'

        cursor.execute(query)
        answer_id = [{'audio' : person_id + '_' + c[1], 'video' : person_id + '_' + c[2], 'answer' : c[3], 'text' : text} for c in cursor][0]
        print(answer_id['answer'])
        return answer_id

    else:
        answer = '시민님, 적절한 답변이 없습니다.'
        return {'answer' : answer, 'text' : text}

@app.route('/save-record', methods=['POST'])
def save_record():
    # check if the post request has the file part // 녹음 존재 여부 확인
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file_name = 'Temp' + ".wav"
    full_file_name = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    file.save(full_file_name)
    emb_df = dfs[category_dict[category]]
    text = infer_STT(full_file_name)
    
    if toxic_check(text): # 욕설 감지
        ment = "욕설이 감지되었습니다."
        return ment
    
    print(text)

    top_k_cross_scores, top_k_indices = infer_POLY(text, emb_df)
    max_idx = np.argmax(top_k_cross_scores)

    if top_k_cross_scores[max_idx] > 10:
        emb_idx = top_k_indices[max_idx]
        answer_id = emb_df.iloc[emb_idx]['index']
        print(answer_id)

        cursor = db_conn.cursor()

        query = f'select * from path_table where id = {answer_id}'

        cursor.execute(query)
        answer_id = [{'audio' : person_id + '_' + c[1], 'video' : person_id + '_' + c[2], 'answer' : c[3], 'text' : text} for c in cursor][0]
        print(answer_id['answer'])
        return answer_id

    else:
        answer = '시민님, 적절한 답변이 없습니다.'
        return {'answer' : answer, 'text' : text}
    
if __name__ == '__main__':
    # app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    app.run(host='0.0.0.0', port='8080', debug=True, threaded=True, use_reloader=False)