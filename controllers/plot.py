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
values_dict = [0, 5, 3, 3, 3, 4, 2, 2]


img_list = []
IMG_SIZE = 100


def image_plot():
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

    score_list = []

    for i in prediction_list:
        vio_score = values_dict[i]
        score_list.append(vio_score)

    print(score_list)
    return score_list




