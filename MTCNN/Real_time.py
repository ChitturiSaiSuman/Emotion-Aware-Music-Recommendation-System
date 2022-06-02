from fer import FER
import cv2
from sys import argv
from os import listdir
import time

detector = FER()


def save_image(array):
    cv2.imwrite("Capture.jpg", array)


def get_emotion(img_path: str) -> str:
    try:
        img = cv2.imread(img_path)
        emotions = detector.detect_emotions(img)
        # print(emotions)
        weights = emotions[0]["emotions"]
        weights = [(weights[emotion], emotion) for emotion in weights]
        weights.sort()
        return weights[-1][1]
    except:
        return "No face"


def real_time():
    arg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_haar_cascade = cv2.CascadeClassifier(arg)
    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()
        if not ret:
            continue
        cv2.imwrite("Capture.jpg", test_img)
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

            predicted_emotion = get_emotion("Capture.jpg")

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

        time.sleep(0.033)

    cap.release()
    cv2.destroyAllWindows


real_time()
