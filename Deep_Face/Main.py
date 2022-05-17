import cv2
from deepface import DeepFace
import os

def get_emotion(img) -> str:
    try:
        result = DeepFace.analyze(img, actions=['emotion'])
        return result['dominant_emotion']
    except:
        return 'No face detected'

def from_file(img_path: str) -> str:
    array = cv2.imread(img_path)
    emotion = get_emotion(array)
    return emotion

def validate():
    path = "../Datasets/Facial Emotion Recognition/AffectNet/Annotated/images/"
    labels = os.listdir(path)
    total_correct = 0
    total_present = 0
    for label in labels:
        emotion_correct = 0
        emotion_total = 0
        if label == 'Contempt':
            continue

        actual_emotion = label.lower()
        if actual_emotion == 'anger':
            actual_emotion = 'angry'

        print("Currently Testing " + str(label), flush = True)
        print(flush = True)

        for img in os.listdir(path + label):
            emotion = from_file(path + label + '/' + img)
            if emotion.lower() == actual_emotion:
                emotion_correct += 1
                total_correct += 1
            emotion_total += 1
            total_present += 1

            if total_present % 1000 == 0:
                print("Processed " + str(total_present) + " images so far", flush = True)
                print("Current Accuracy: " + str(total_correct / total_present), flush = True)
                print(flush = True)
        
        print("Accuracy for " + label + ": " + str(emotion_correct / emotion_total), flush = True)
        print("Details: " + str(emotion_correct) + "/" + str(emotion_total), flush = True)
        print(flush = True)

    print("Total Accuracy: " + str(total_correct / total_present), flush = True)
    print("Total Correct: " + str(total_correct), flush = True)
    print("Total Present: " + str(total_present), flush = True)
    print(flush = True)


if __name__ == '__main__':
    validate()