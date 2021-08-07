import tensorflow
import cv2
import pandas
import numpy as np
import os
import tqdm
from collections import Counter

img_path = './data/image_data/'
model_path = './models/new_image_model.h5'
CATEGORIES = ["non-violence", "blood", "bomb", "explosion", "fighting", "gunshoting", "hitting", "knife"]
values_dict = [3, 5, 3, 3, 3, 4, 2, 2]


img_list = []
IMG_SIZE = 100


def image():
    img_dir = os.listdir(img_path)
    for img in img_dir:

        try:

            img_array = cv2.imread(os.path.join(img_path, img), cv2.IMREAD_GRAYSCALE)

            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

            img_list.append(new_array)
        except Exception as e:
            pass

    prediction_list = []
    cleaned_list = []
    vals = 0

    model = tensorflow.keras.models.load_model(model_path)

    for data in img_list:

        prediction = model.predict(data)
        # print(prediction)
        pred = np.argmax(prediction, axis=1)
        classes = pred[0]
        prediction_list.append(classes)
    print(prediction_list)
    counter = Counter(prediction_list)
    new_cls = sorted(counter.items())

    for i in range(0, (len(new_cls))):
        val = new_cls[i][1]
        cleaned_list.append(val)

    nonv_score = cleaned_list[0] * values_dict[0]

    for j in range(1, len(cleaned_list)):
        val = cleaned_list[j] * values_dict[j]
        vals = vals + val

    full_score = nonv_score + vals
    full_length = len(prediction_list)
    #print(full_score)

    violent_score = ((full_score - nonv_score) / full_score) * 100

    #print("Violence precentage is: {} %".format("%.2f" % violent_score))
    rounded_final_score = round(violent_score)
    val_frames = (full_length - cleaned_list[0])

    # val_frames = rounded_final_score * 1.5
    # return None
    return violent_score, val_frames, full_length
    # print("Violence precentage is: {} %".format("%.2f" % final_score))






