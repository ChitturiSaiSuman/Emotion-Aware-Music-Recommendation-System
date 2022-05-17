import os
import re
import shutil
import numpy as np
from collections import defaultdict

exp = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Surprise",
    4: "Fear",
    5: "Disgust", 
    6: "Anger",
    7: "Contempt"
}

exp_files = filter(lambda x: 'exp' in x, os.listdir('train_set/annotations/'))

count = defaultdict(int)

for file in exp_files:
    data = np.load('train_set/annotations/' + file)
    data = int(str(data))
    expression = exp[data]

    id = re.findall(r'\d+', file)[0]

    count[expression] += 1
    
    image_file = 'train_set/images/' + str(id) + '.jpg'
    os.rename(image_file, 'train_set/images/' + expression + '_' + str(count[expression]).zfill(7) + '.jpg')