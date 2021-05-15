import pandas as pd
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import Memory
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import scatter

wine = load_wine()
features, target = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(features, target)

scaler = StandardScaler().fit(X_train)
X_test_std = scaler.transform(X_test)
X_train_std = scaler.transform(X_train)

dummy_clf = DummyClassifier()

# cache_dir = os.path.join('cache')
# memory = Memory(cache_dir)


dummy_clf.fit(X_test_std, y_test)
print(f"dummy TEST score: {dummy_clf.score(X_test_std, y_test)}")

dummy_clf.fit(X_train_std, y_train)
print(f"dummy TRAIN score: {dummy_clf.score(X_train_std, y_train)}")

k_test = {}
k_train = {}

for i in range(1, 21):
    kneighbor_clf = KNeighborsClassifier(n_neighbors=i)
    kneighbor_clf.fit(X_test_std, y_test)
    k_test[i] = kneighbor_clf.score(X_test_std, y_test)
    kneighbor_clf.fit(X_train_std, y_train)
    k_train[i] = kneighbor_clf.score(X_train_std, y_train)

for key, value in k_test.items():
    print(f"Kneighbor TEST score for {key} neighbors: {value}")

for key, value in k_train.items():
    print(f"Kneighbor TRAIN score for {key} neighbors: {value}")

x, y = k_test.keys(), k_test.values()
x1, y1 = k_train.keys(), k_train.values()
scatter(x, y, c="#7f7f7f")
scatter(x1, y1)
plt.legend(['test', 'train'])
plt.show()
