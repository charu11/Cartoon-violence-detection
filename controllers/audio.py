import os
import numpy as np
import librosa
from pydub import AudioSegment
from mutagen.wave import WAVE
import matplotlib.pyplot as plt
from collections import Counter
import cv2
import tensorflow

audio_dir = './data/audio_data/audio.wav'
split_dir = './data/audio_data/split_audio/'
spectogram_dir = './data/audio_data/spectogram/'
model_path = './models/new_audio_image_model.h5'

CATEGORIES = ['Non-Violence', "Explosion", "Fight_sound", "Fire", "knife", "Gunshots", "Scary", "Scream"]
values_dict = [3, 5, 3, 3, 3, 4, 2, 2]
IMG_SIZE = 100
spec_list = []


def audio():

    audio_WAVE = WAVE(audio_dir)
    audio_info = audio_WAVE.info
    length = int(audio_info.length)
    audio_pieces = round(length / 5)

    print(audio_pieces)

    for i in range(1, audio_pieces-1):
        i = i * 5000
        j = (i + 1) * 5000

        new_audio = AudioSegment.from_wav(audio_dir)
        new_audio = new_audio[i: j]
        new_audio.export('./data/audio_data/split_audio/audio' + str(i) + '.wav', format='wav')

    cmap = plt.get_cmap('inferno')

    for file in os.listdir(split_dir):
        audio_name = os.path.join(split_dir, file)
        y, sr = librosa.load(audio_name, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off')
        plt.savefig(f'./data/audio_data/spectogram/{file[:-3].replace(".", "")}.png')
        plt.clf()

    img_dir = os.listdir(spectogram_dir)
    for img in img_dir:

        try:

            img_array = cv2.imread(os.path.join(spectogram_dir, img), cv2.IMREAD_GRAYSCALE)

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

            spec_list.append(new_array)

        except Exception as e:
            pass

    prediction_list = []
    classes = []
    cleaned_list = []
    vals = 0
    nonv_score = 0

    model = tensorflow.keras.models.load_model(model_path)

    for data in spec_list:

        prediction = model.predict(data)
        # print(prediction)
        pred = np.argmax(prediction, axis=1)
        classes = pred[0]
        prediction_list.append(classes)

    counter = Counter(prediction_list)
    new_cls = sorted(counter.items())
    num = new_cls[0][0]

    for i in range(0, (len(new_cls))):
        val = new_cls[i][1]
        cleaned_list.append(val)

    for j in range(0, len(cleaned_list)):
        val = cleaned_list[j] * values_dict[j]
        vals += val

    if num != 0:
        full_score = vals
    else:
        nonv_score = cleaned_list[0] * values_dict[0]
        full_score = nonv_score + vals


    full_length = len(prediction_list)
    # print(full_score)

    violent_score = ((full_score - nonv_score) / full_score) * 100
    print(nonv_score)
    print(vals)
    print(prediction_list)
    print(new_cls[0][0])
    print(violent_score)

    return violent_score
    # print("Violence precentage is: {} %".format("%.2f" % final_score))


audio()

