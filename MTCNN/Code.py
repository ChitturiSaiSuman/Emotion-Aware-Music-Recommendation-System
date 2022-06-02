from fer import FER
import cv2
from sys import argv
from os import listdir

path = (
    "../Datasets/Facial Emotion Recognition/AffectNet/Annotated/images/"
)


def get_emotion(img_path: str) -> str:
    try:
        img = cv2.imread(img_path)
        detector = FER()
        emotions = detector.detect_emotions(img)
        # print(emotions)
        weights = emotions[0]["emotions"]
        weights = [(weights[emotion], emotion) for emotion in weights]
        weights.sort()
        return weights[-1][1]
    except:
        return "No face"


def main():
    labels = listdir(path)

    total_correct = 0
    total_present = 0

    for label in labels:
        if label == "Contempt":
            continue
        else:
            label_correct = 0
            label_present = 0
            print("Testing:", label, flush=True)
            for image in listdir(path + label):
                got = get_emotion(path + label + "/" + image)
                expected = label.lower()
                if expected == "anger":
                    expected = "angry"
                if got == expected:
                    label_correct += 1
                    total_correct += 1
                label_present += 1
                total_present += 1
            accuracy = label_correct / label_present
            print("Accuracy for", label, ":", accuracy, flush=True)
            print("Label Present:", label_present, flush=True)
            print("Label Correct:", label_correct, flush=True)
            print("", flush=True)
    print(
        "Total Accuracy:",
        total_correct / total_present,
        flush=True,
    )
    print("Total Present:", total_present, flush=True)
    print("Total Correct:", total_correct, flush=True)


main()
