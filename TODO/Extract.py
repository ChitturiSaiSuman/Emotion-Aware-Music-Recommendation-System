from collections import defaultdict
from sys import stderr
from re import findall


headings = input().split(',')
data = []

coordinates = {
    'anger': (2.50, 5.93, 5.14),
    'happy': (8.21, 5.55, 7.00),
    'surprise': (7.21, 7.54, 7.25),
    'disgust': (1.69, 3.33, 4.46),
    'fear': (2.97, 5.16, 2.87),
    'sad': (2.40, 2.81, 3.84),
    'neutral': (4.12, 3.38, 4.43)
}

freq = defaultdict(int)
first = defaultdict(str)

while True:
    try:
        line = input()
        data.append(line.split(','))
    except:
        break

def get_emotion(x, y, z):
    def distance(x1, y1, z1, x2, y2, z2):
        # returns manhattan distance
        # return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
    min_distance = 10**10
    closest_emotion = None
    for emotion in coordinates:
        x2, y2, z2 = coordinates[emotion]
        distance_to_emotion = distance(x, y, z, x2, y2, z2)
        if distance_to_emotion < min_distance:
            min_distance = distance_to_emotion
            closest_emotion = emotion
    return closest_emotion

# min_valence = 10**10
# min_arousal = 10**10
# min_dominance = 10**10

# max_valence = 0
# max_arousal = 0
# max_dominance = 0

# max_tuple = (0, 0, 0)
# min_tuple = (10, 10, 10)

for i in range(len(data)):
    try:
        valence, arousal, dominance = map(float, [data[i][-6], data[i][-5], data[i][-4]])
    except:
        print(i)

    # max_tuple = max(max_tuple, (valence, arousal, dominance))
    # min_tuple = min(min_tuple, (valence, arousal, dominance))

    # max_valence = max(max_valence, valence)
    # max_arousal = max(max_arousal, arousal)
    # max_dominance = max(max_dominance, dominance)

    # min_valence = min(min_valence, valence)
    # min_arousal = min(min_arousal, arousal)
    # min_dominance = min(min_dominance, dominance)

    # print(valence, arousal, dominance, file = stderr)
    # if any([item < 0 for item in [valence, arousal, dominance]]):
    #     print('Negative found', file = stderr)
    emotion = get_emotion(valence, arousal, dominance)
    if emotion not in first:
        first[emotion] = data[i][0]
    freq[emotion] += 1
    data[i].append(emotion)

print(freq)
print(first)


# print(max_valence, max_arousal, max_dominance)
# print(min_valence, min_arousal, min_dominance)
# print(max_tuple)
# print(min_tuple)

# print(','.join(headings))
# for line in data:
#     print(','.join(line))

# print(type(data[0]))

# distinct = set()

# for i in range(len(data)):
#     data[i] = findall(r'\[(.*?)\]', data[i])
#     data[i] = data[i][0].split(', ')
#     data[i] = [item[1:len(item) - 1] for item in data[i]]
#     distinct.add(tuple(data[i]))

# print(distinct)
# for 