import cv2

def save_image(array):
    cv2.imwrite("Capture.jpg", array)

def capture_image():
    cap_obj = cv2.VideoCapture(0)
    status, img = cap_obj.read()
    if not status:
        raise AssertionError("Unable to capture")
    return img

save_image(capture_image())