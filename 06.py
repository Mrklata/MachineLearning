import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist


class Analytyst:
    def __init__(self) -> None:
        self.seed = np.random.seed(61)
        self.x, self.y = make_blobs(
            n_samples=200,
            centers=4,
            cluster_std=0.5,
            shuffle=True,
            random_state=self.seed
        )
        self.km_classifier = KMeans(
            n_clusters=4,
            init='random',
            n_init=10,
            max_iter=300,
            random_state=self.seed
        )
        self.y_km = self.km_classifier.fit_predict(self.x)
        self.x_wine, self.y_wine = load_wine(return_X_y=True)
        x_train, x_test, y_train, y_test = train_test_split(self.x_wine, self.y_wine, test_size=0.2,
                                                            random_state=self.seed)
        scaler = StandardScaler().fit(x_train)
        x_train_std = scaler.transform(x_train)
        self.pca = PCA(n_components=2, whiten=True)
        self.x_reducted = self.pca.fit_transform(x_train_std)

    def blobs(self) -> None:

        plt.figure(figsize=(10, 8))
        plt.scatter(
            self.x[:, 0],
            self.x[:, 1],
            s=100
        )
        plt.show()

    def k_means(self) -> None:
        print(self.km_classifier.cluster_centers_)

    def ploting_scatter(self) -> None:
        plt.scatter(
            self.x[self.y_km == 0, 0],
            self.x[self.y_km == 0, 1],
            c='red', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.x[self.y_km == 1, 0],
            self.x[self.y_km == 1, 1],
            c='green', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.x[self.y_km == 2, 0],
            self.x[self.y_km == 2, 1],
            c='blue', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.x[self.y_km == 3, 0],
            self.x[self.y_km == 3, 1],
            c='orange', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.km_classifier.cluster_centers_[:, 0],
            self.km_classifier.cluster_centers_[:, 1],
            c='yellow', marker="o", s=200, alpha=0.6,
            edgecolor="black", label='Cent'
        )

        plt.legend()
        plt.grid()
        plt.show()

    def std_scaling(self) -> None:

        print(self.pca.explained_variance_ratio_)
        print(self.pca.singular_values_)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.x_reducted[:, 0],
            self.x_reducted[:, 1],
            s=100
        )

        plt.show()

    def elbow(self) -> None:
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.x_wine)
            distortions.append(
                sum(np.min(cdist(self.x_wine, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.x_wine.shape[
                    0])

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('Elbow Method: optimal k')
        plt.show()

    def silhouette(self) -> None:
        range_n_clusters = list(range(2, 10))
        silhouette_score = []

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters, random_state=self.seed)
            cluster_labels = clusterer.fit_predict(self.x_wine)

            silhouette_avg = metrics.silhouette_score(self.x_wine, cluster_labels)
            silhouette_score.append(silhouette_avg)

            print(f"AVG silhouette_score: {silhouette_avg} for n_clusters: {n_clusters}")

        plt.plot(range_n_clusters, silhouette_score, 'bx-')
        plt.xlabel('k')
        plt.ylabel('silhouette_score')
        plt.title('Silhouette Coefficient method: the optimal k')
        plt.show()

    def plot_second_scatter(self) -> None:
        y_km = self.km_classifier.fit_predict(self.x_reducted)
        plt.figure(figsize=(10, 8))

        plt.scatter(
            self.x_reducted[y_km == 0, 0],
            self.x_reducted[y_km == 0, 1],
            c='green', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.x_reducted[y_km == 1, 0],
            self.x_reducted[y_km == 1, 1],
            c='orange', edgecolor='black',
            label='1', s=80
        )

        plt.scatter(
            self.km_classifier.cluster_centers_[:, 0],
            self.km_classifier.cluster_centers_[:, 1],
            c='r', marker="o", s=200, alpha=0.6,
            edgecolor="black", label='Cent'
        )

        plt.legend()
        plt.grid()
        plt.show()

    def k_mean(self) -> None:
        for i in ["random", "k-means++"]:
            k_mean = KMeans(
                n_clusters=2,
                init=i,
                n_init=10,
                max_iter=300,
                random_state=self.seed
            )

            k_mean.fit(self.x_wine, self.y_wine)
            prediction = k_mean.predict(self.x_wine)
            print(f"init: {i}, accuracy: {accuracy_score(self.y_wine, prediction)}")

    def mean_shift(self) -> None:
        bandwidth = estimate_bandwidth(self.x_wine, quantile=0.2, n_samples=500)
        mean_s = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        mean_s.fit(self.x_wine)
        labels = mean_s.labels_
        cluster_centers = mean_s.cluster_centers_

        labels_unique = set(labels)

        clusters = len(labels_unique)
        return clusters, cluster_centers, labels

    def plot(self) -> None:
        plt.figure(figsize=(10, 8))
        plt.clf()
        clusters, cluster_centers, labels = self.mean_shift()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(clusters), colors):
            members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(self.x_wine[members, 0], self.x_wine[members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title(f'clusters: {clusters}')
        plt.show()