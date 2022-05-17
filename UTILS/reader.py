import numpy as np
import os

'0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt'

files = os.listdir('train_set/annotations/')[:100]

for file in files:
    data = np.load('train_set/annotations/' + file)
    if 'exp' in file:
        print(file, data)