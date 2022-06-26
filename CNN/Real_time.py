import cv2
import numpy as np
from keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from IPython.display import clear_output
import warnings

warnings.filterwarnings("ignore")

test_dir = "temp"

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

preprocess_fun = tf.keras.applications.densenet.preprocess_input

model = load_model("best_model.h5")

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    preprocessing_function=preprocess_fun,
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    color_mode="rgb",
    class_mode="categorical",
    seed=12,
)

model.evaluate(test_generator)
preds = model.predict(test_generator)

print(preds)

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(
        gray_img, 1.32, 5
    )

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(
            test_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            thickness=7,
        )
        roi_gray = gray_img[y : y + w, x : x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        print(predictions)

        max_index = np.argmax(predictions[0])

        emotions = (
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        )
        predicted_emotion = emotions[max_index]

        cv2.putText(
            test_img,
            predicted_emotion,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow("Facial emotion analysis ", resized_img)

    if cv2.waitKey(10) == ord("q"):
        break

    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows
