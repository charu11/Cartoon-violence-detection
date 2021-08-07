import tensorflow
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from collections import Counter
import boto3
from botocore.exceptions import NoCredentialsError
import time
import urllib
import urllib.request
import json
import tscribe
import re
from collections import Counter
from datetime import datetime

import logging
from botocore.exceptions import ClientError

date = datetime.now()
audio_dir = '../data/audio_data/audio.wav'
corpus_dir = './data/text_data/corpus.csv'
model_path = './models/text_model.h5'

ACCESS_KEY = 'AKIAWA4UJTCPBLENP3UA'
SECRET_KEY = '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'
bucket = 'videoviolence'
s3_file = 'audio.wav'
max_length = 100

corpus = pd.read_csv(corpus_dir)
corpus_dict = dict(zip(corpus.Words, corpus.Values))





'''

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
    
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

'''


def transcribe():
    #

    job_name = 'videoviolence124'
    job_uri = 's3://videoviolence/audio.wav'
    transcribe_audio = boto3.client('transcribe', aws_access_key_id=ACCESS_KEY,
                                    aws_secret_access_key = SECRET_KEY, region_name='us-east-1')

    #transcribe_audio.start_transcription_job(TranscriptionJobName=job_name, Media={'MediaFileUri': job_uri}, MediaFormat='wav', LanguageCode='en-US')

    while True:
        status = transcribe_audio.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("Not ready yet...")
        time.sleep(5)
        #print(status)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        print('job completed')
        response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        text = data['results']['transcripts'][0]['transcript']
        confidence = data['results']['items'][0]['alternatives'][0]['confidence']
        #print("data", data)
        print(text)
        return text


word_list = []
vals_array = []


def split():
    full_words = transcribe()
    #full_words = ' '.join(full_words)
    sentence = re.split(r'(?<=\w\.)\s', full_words)
    for i in sentence:
        word_list.append(i)
    #print(word_list)
    return word_list


def text():
   # upload_to_aws(audio_dir, bucket)
    words = split()
    df = pd.DataFrame(words, columns=['text'])
    model = tensorflow.keras.models.load_model(model_path)

    token = Tokenizer()
    token.fit_on_texts(df['text'])
    seq = token.texts_to_sequences(df['text'])
    pad_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(pad_seq)
    classes = np.argmax(pred, axis=1)
    classes = classes.tolist()
    df['value'] = pd.DataFrame(classes)
    pos_df = df[df['value'] == 1]['text']

    my = pos_df.to_list()
    my_str = ' '.join(my)
    word_split = my_str.split()
    print(word_split)

    for i in word_split:
        if i in corpus_dict.keys():
            vals = corpus_dict.get(i)
            vals_array.append((vals))

    print(vals_array)
    val_score = sum(vals_array)
    print(val_score)

    non_val_score = (len(word_split) - len(vals_array)) * 3
    print(non_val_score)

    full_score = non_val_score + val_score
    print(full_score)

    val_per = (val_score / full_score) * 100

    # val_soft = softmax(val_score)
    print(val_per)

    return val_per








# upload_to_aws(audio_dir, bucket, s3_file)
# upload_to_aws(audio_dir, bucket, s3_file)
words = text()
print(words)

