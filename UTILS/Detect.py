import cv2
import matplotlib.pyplot as plt
import os
from deepface import DeepFace
from numpy import real

def read_all():
    for file in os.listdir():
        if '.jpeg' in file or '.jpg' in file:
            img = cv2.imread(file)
            plt.imshow(img[:,:,::-1])
            # plt.show()

            result = DeepFace.analyze(img, actions=['emotion'])

            print(file, result['dominant_emotion'])

def real_time():
    pass

def read_image():

    file = 'Capture.jpg'
    img = cv2.imread(file)

    result = DeepFace.analyze(img, actions=['emotion'])

    print(file, result['dominant_emotion'])

def main():
    read_image()

main()