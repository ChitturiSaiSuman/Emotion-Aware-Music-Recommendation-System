cap_obj = cv2.VideoCapture(0)

def capture_image():
    status, img = cap_obj.read()
    if not status:
        raise AssertionError("Unable to capture")
    return img

def merge_dicts(dict1: dict, dict2: dict) -> None:
    for i in dict2:
        if i not in dict1:
            dict1[i] = dict2[i]
        else:
            dict1[i] += dict2[i]

def from_file(path: str):
    freq = dict()

    for i in range(10):
        array = cv2.imread(path)

        emotion = get_emotion(array)
        if emotion != ['No face detected']:
            merge_dicts(freq, emotion['emotion'])
            emotion = max([(freq[i], i) for i in freq])[-1]
        faces_detected = face_haar_cascade.detectMultiScale(array, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(array, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            cv2.putText(array, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(array, (800, 600))
        cv2.imshow('Facial emotion analysis ', array)
        if cv2.waitKey(10) == ord('q'):
            break
        time.sleep(0.01)
    
    if freq != {}:
        freq = [(freq[i], i) for i in freq]
        freq.sort()
        print(freq)
        print(freq[-1][-1])

def realtime():
    freq = dict()
    for i in range(30):
        array = capture_image()
        emotion = get_emotion(array)
        if emotion != ['No face detected']:
            merge_dicts(freq, emotion['emotion'])
            emotion = max([(freq[i], i) for i in freq])[-1]
        try:
            faces_detected = face_haar_cascade.detectMultiScale(array, 1.32, 5)
        except:
            print("No face detected")

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(array, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            cv2.putText(array, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(array, (500, 350))
        cv2.imshow('Facial emotion analysis ', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break
        time.sleep(0.001)

    freq = [(freq[i], i) for i in freq]
    freq.sort()
    print(freq)
    print(freq[-1][-1])

def de_flatten(arr: list, m: int, n: int) -> list:
    arr = array(arr)[indices.astype(int)]
    mat = []
    row = []
    temp = []
    for i in arr:
        temp.append(i)
        if len(temp) == 3:
            temp = array(temp, dtype=uint8)
            row.append(temp)
            temp = []

        if len(row) == 32:
            mat.append(ndarray(row))
            row = []
    return ndarray(mat)

def test():
    print("Parsing CSV")
    train, public, private = parse('fer2013.csv')

    print("Cleaning")
    for i in range(len(public)):
        public[i] = (public[i][0], list(map(int, public[i][1].split())))
    for i in range(len(private)):
        private[i] = (private[i][0], list(map(int, private[i][1].split())))

    for i in range(len(public)):
        public[i] = (public[i][0], de_flatten(public[i][1], 48, 48))
    for i in range(len(private)):
        private[i] = (private[i][0], de_flatten(private[i][1], 48, 48))

    labels = {
        '0': 'angry',
        '1': 'disgust',
        '2': 'fear',
        '3': 'happy',
        '4': 'sad',
        '5': 'surprise',
        '6': 'neutral'
    }
    # reverse_labels = {labels[i]: i for i in labels}

    overall_correct = 0
    overall_total = 0

    correct = 0
    total = 0

    print("Testing Public")
    for data in public:
        label, mat = data
        emotion_got = get_emotion(mat)
        print(emotion_got)
        emotion_expected = labels[label]
        if emotion_got == emotion_expected:
            correct += 1
        total += 1

    print("Public Accuracy: ", (correct / total))

    overall_correct += correct
    overall_total += total

    correct = 0
    total = 0

    print("Testing Private")
    for data in private:
        label, mat = data
        emotion_got = get_emotion(mat)
        print(emotion_got)
        emotion_expected = labels[label]
        if emotion_got == emotion_expected:
            correct += 1
        total += 1

    print("Private Accuracy: ", (correct / total))

    overall_correct += correct
    overall_total += total

    print("Total Accuracy: ", overall_correct / overall_total)