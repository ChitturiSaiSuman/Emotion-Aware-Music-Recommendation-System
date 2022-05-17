import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings('ignore')


SEED = 12
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 64
EPOCHS = 30
FINE_TUNING_EPOCHS = 20
LR = 0.01
NUM_CLASSES = 7
EARLY_STOPPING_CRITERIA=3
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]

model = tf.keras.models.load_model('best_model.h5')

def get_emotion(path: str) -> str:
    preprocess_fun = tf.keras.applications.densenet.preprocess_input
    test_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2, preprocessing_function = preprocess_fun)
    test_generator = test_datagen.flow_from_directory(directory = path, target_size = (IMG_HEIGHT,IMG_WIDTH), batch_size = BATCH_SIZE, shuffle = False, color_mode = "rgb", class_mode = "categorical", seed = 12)
    results = model.predict(test_generator)
    freq = [0] * 7
    for result in results:
        freq[np.argmax(result)] += 1
    
    return CLASS_LABELS[np.argmax(freq)]


def real_time():
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    snapped = 0
    while snapped < 150:
        ret, test_img = cap.read()
        if ret:
            snapped += 1
            cv2.imwrite("realtime/stream/cap_" + str(snapped) + ".jpg", test_img)

    print("Detecting the Emotion...", flush = True)

    emotion = get_emotion('data/')
    print("Emotion: ", emotion)

    cap.release()
    cv2.destroyAllWindows

real_time()
    
    # For Preview

    # gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    # for (x, y, w, h) in faces_detected:
    #     cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
    #     roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
    #     roi_gray = cv2.resize(roi_gray, (48, 48))
    #     img_pixels = image.img_to_array(roi_gray)
    #     img_pixels = np.expand_dims(img_pixels, axis=0)
    #     img_pixels /= 255

    #     predictions = model.predict(img_pixels)
    #     print(predictions)

    #     # find max indexed array
    #     max_index = np.argmax(predictions[0])

    #     emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
    #     predicted_emotion = emotions[max_index]

    #     cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # resized_img = cv2.resize(test_img, (1000, 700))
    # cv2.imshow('Facial emotion analysis ', resized_img)

    # if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
    #     break

    # time.sleep(0.1)