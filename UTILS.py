import numpy, pandas, collections, fer, random
from sklearn.cluster import KMeans

constants = {
    'Music_Dataset': 'Datasets/Music Recommendation/muse_v3.csv',

    'emotion_model': fer.FER(),

    'frame': """<iframe src="https://open.spotify.com/embed/track/{}" width="260" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>""",
    
    'coordinates': {
        'angry': (2.50, 5.93, 5.14),
        'happy': (8.21, 5.55, 7.00),
        'surprise': (7.21, 7.54, 7.25),
        'disgust': (1.69, 3.33, 4.46),
        'fear': (2.97, 5.16, 2.87),
        'sad': (2.40, 2.81, 3.84),
        'neutral': (4.12, 3.38, 4.43)
    },

    'colors': {
        'angry': '#ff3300',
        'happy': '#007399',
        'surprise': '#e13a37',
        'disgust': '#663300',
        'fear': '#666699',
        'sad': '#6b6b47',
        'neutral': '#862d2d',
        'left': 'black',
    }
}

def detect_emotion(img: numpy.ndarray) -> str:
    try:
        emotion_detection_model = constants['emotion_model']
        emotions = emotion_detection_model.detect_emotions(img)
        weights = emotions[0]['emotions']
        weights = [(weights[emotion], emotion) for emotion in weights]
        weights.sort()
        return weights[-1][1]
    except:
        return None

def distance(x1: float, y1: float, z1: float,
             x2: float, y2: float, z2: float) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def map_emotion(valence: float, arousal: float, dominance: float) -> tuple:
    min_distance = 10**10
    closest_emotion = None
    for emotion in constants['coordinates']:
        x2, y2, z2 = constants['coordinates'][emotion]
        distance_to_emotion = distance(valence, arousal, dominance, x2, y2, z2)
        if distance_to_emotion < min_distance:
            min_distance = distance_to_emotion
            closest_emotion = emotion
    return (closest_emotion, min_distance)

def get_centroids() -> tuple:
    coordinates = constants['coordinates']
    coordinates = [(key, value) for key, value in coordinates.items()]
    centroids = [list(item[1]) for item in coordinates]
    centroids = numpy.array(centroids)
    labels = [item[0] for item in coordinates]
    return (labels, centroids)

def pre_process_static() -> dict:
    dataframe = pandas.read_csv(constants['Music_Dataset'])
    dataframe = dataframe[['track', 'artist', 'genre', 'spotify_id',
                            'valence_tags', 'arousal_tags', 'dominance_tags']]
    
    tracks = collections.defaultdict(list)

    for index, row in dataframe.iterrows():
        valence, arousal, dominance = map(float, [row['valence_tags'], row['arousal_tags'], row['dominance_tags']])
        spotify_id = row['spotify_id']
        if pandas.notna(spotify_id):
            emotion, distance = map_emotion(valence, arousal, dominance)
            tracks[emotion].append((spotify_id, distance))

    for track in tracks:
        tracks[track].sort(key = lambda x: x[-1])
        tracks[track] = [item[0] for item in tracks[track]]

    return tracks

def pre_process_cluster() -> dict:
    data = pandas.read_csv(constants['Music_Dataset'])
    data = data[data['spotify_id'].notna()]
    labels, centroids = get_centroids()
    x = data.iloc[:, 5:8].values
    k_means_optimum = KMeans(n_clusters = 7, n_init = 1, init = centroids,  random_state = 50, tol = 1e-8)
    y = k_means_optimum.fit_predict(x)
    data['cluster'] = y
    tracks = collections.defaultdict(list)
    for index, row in data.iterrows():
        spotify_id = row['spotify_id']
        cluster = row['cluster']
        tracks[labels[cluster]].append(spotify_id)

    return tracks

# Uncomment the following line for 
# static pre-processing
# Static pre-processing uses defined coordinates

# tracks = pre_process_static()

# The following relies on KMeans
# Clustering which gives better results

tracks = pre_process_cluster()

def get_top_k(emotion: str, k = 15, sample_size = 100) -> list:
    top_k = random.sample(tracks[emotion][:sample_size], k)
    return top_k