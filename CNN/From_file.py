import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")
from keras.models import load_model
import numpy as np
import time

model = load_model("best_model_so_far.h5")


face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

labels = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

cap = cv2.VideoCapture(0)

for file in os.listdir("data"):
    file = "data/" + file
    test_img = cv2.imread(file)

    actual_emotion = ""
    for emotion in labels:
        if emotion.lower() in file.lower():
            actual_emotion = emotion

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

    time.sleep(3)

cap.release()
cv2.destroyAllWindows
