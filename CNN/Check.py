import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix , classification_report 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')
import cv2
from os import listdir
import shutil

SEED = 12
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 64
EPOCHS = 30
FINE_TUNING_EPOCHS = 20
LR = 0.01
NUM_CLASSES = 7
EARLY_STOPPING_CRITERIA=3
CLASS_LABELS  = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', "Surprise"]

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

def main():
    path = '../Datasets/Facial Emotion Recognition/AffectNet/Annotated/images/'
    labels = listdir(path)

    total_correct = 0
    total_present = 0

    for label in labels:
        if label == 'Contempt':
            continue
        else:
            destination = "Stream/Stream/"

            label_correct = 0
            label_present = 0
            print("Testing:", label,flush = True, file=open('LOG.txt', 'a'))
            for image in listdir(path + label):
                shutil.copy(path + label + "/" + image, destination + "cap.jpg")
                got = get_emotion("Stream")
                expected = label
                if got.lower() == expected.lower():
                    label_correct += 1
                    total_correct += 1
                label_present += 1
                total_present += 1
            print("Accuracy for", label, ":", label_correct / label_present,flush = True, file=open('LOG.txt', 'a'))
            print("Label Present:", label_present,flush = True, file=open('LOG.txt', 'a'))
            print("Label Correct:", label_correct,flush = True, file=open('LOG.txt', 'a'))
            print("",flush = True, file=open('LOG.txt', 'a'))
    print("Total Accuracy:", total_correct / total_present,flush = True, file=open('LOG.txt', 'a'))
    print("Total Present:", total_present,flush = True, file=open('LOG.txt', 'a'))
    print("Total Correct:", total_correct,flush = True, file=open('LOG.txt', 'a'))

main()