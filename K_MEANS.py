import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy, pandas
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import UTILS

constants = UTILS.constants

def get_data_frame() -> pandas.DataFrame:
    print("Parsing Dataset...")
    path = constants['Music_Dataset']
    dataframe = pandas.read_csv(path)
    dataframe = dataframe[dataframe['spotify_id'].notna()]
    return dataframe

def get_static_centroids() -> tuple:
    coordinates = constants['coordinates']
    coordinates = [(key, value) for key, value in coordinates.items()]
    centroids = [list(item[1]) for item in coordinates]
    centroids = numpy.array(centroids)
    labels = [item[0] for item in coordinates]
    return (labels, centroids)

def split_dataframe(dataframe: pandas.DataFrame, train_size = 0.60):
    train_percent = int(train_size * 100)
    test_percent = 100 - train_percent
    print("Splitting Dataframe into Train ({} percent) and Test ({} percent) Set...".format(train_percent, test_percent))
    train_dataframe = dataframe.sample(frac = train_size, random_state = 50)
    test_dataframe = dataframe.drop(train_dataframe.index)
    return (train_dataframe, test_dataframe)

def KMeansAll(dataframe: pandas.DataFrame):
    print("Performing K-Means Clustering without initial centroids...")
    x = dataframe.iloc[:, 5:8].values
    k_means_optimum = KMeans(n_clusters = 7, n_init = 1, init = 'k-means++', random_state = 50, tol = 1e-8)
    y = k_means_optimum.fit_predict(x)
    # dataframe['cluster'] = y
    score = silhouette_score(x, y)
    print("Silhouette score: ", score)

def KMeans_given_Initial_Centroids(dataframe: pandas.DataFrame, centroids: numpy.array):
    print("Performing K-Means Clustering with initial centroids...")
    x = dataframe.iloc[:, 5:8].values
    k_means_optimum = KMeans(n_clusters = 7, n_init = 1, init = centroids, random_state = 50, tol = 1e-8)
    y = k_means_optimum.fit_predict(x)
    # dataframe['cluster'] = y
    score = silhouette_score(x, y)
    print("Silhouette score: ", score)

def KMeans_divided_dataset(whole: pandas.DataFrame, train: pandas.DataFrame, test: pandas.DataFrame):
    print("Performing K-Means Clustering on Train and Test Dataset...")
    x = whole.iloc[:, 5:8].values
    k_means_optimum = KMeans(n_clusters = 7, n_init = 1, init = 'k-means++', random_state = 50, tol = 1e-8)
    y = k_means_optimum.fit_predict(x)
    whole['cluster'] = y
    original_centroids = k_means_optimum.cluster_centers_
    original_centroids = sorted([list(triple) for triple in original_centroids])

    train_x = train.iloc[:, 5:8].values
    k_means_train = KMeans(n_clusters = 7, init = 'k-means++', random_state = 50, tol = 1e-8)
    y_train = k_means_train.fit_predict(train_x)
    train['cluster'] = y_train
    train_centroids = k_means_train.cluster_centers_
    train_centroids = sorted([list(triple) for triple in train_centroids])

    total_correct = 0
    total_present = 0

    for row in test.itertuples():
        valence, arousal, dominance = map(float, [row.valence_tags, row.arousal_tags, row.dominance_tags])
        nearest1 = None
        nearest_distance = 10**10
        for i in range(len(original_centroids)):
            x1, y1, z1 = original_centroids[i]
            x2, y2, z2 = valence, arousal, dominance
            distance_ = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if distance_ < nearest_distance:
                nearest_distance = distance_
                nearest1 = i

        nearest2 = None
        nearest_distance = 10**10
        for i in range(len(train_centroids)):
            x1, y1, z1 = train_centroids[i]
            x2, y2, z2 = valence, arousal, dominance
            distance_ = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            if distance_ < nearest_distance:
                nearest_distance = distance_
                nearest2 = i
        if nearest1 == nearest2:
            total_correct += 1
        total_present += 1

    print("Total Rows in Test Set: ", total_present)
    print("Total Correctly Classified: ", total_correct)
    print("Accuracy: %.3f" %(100 * total_correct / total_present))


if __name__ == '__main__':
    dataframe = get_data_frame()

    # KMeansAll(dataframe.copy(deep = True))

    # labels, centroids = get_static_centroids()
    # KMeans_given_Initial_Centroids(dataframe.copy(deep = True), centroids)

    train_df, test_df = split_dataframe(dataframe.copy(deep = True))
    KMeans_divided_dataset(dataframe.copy(deep = True), train_df, test_df)