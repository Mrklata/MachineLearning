import pandas as pd
import numpy as np
import sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import Memory

wine = load_wine()
features, target = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(features, target)

dummy_clf = DummyClassifier()

# cache_dir = os.path.join('cache')
# memory = Memory(cache_dir)

dummy_clf.fit(features, target)

print(f'dummy TEST score: {dummy_clf.score(X_test, y_test)}')
print(f'dummy TRAIN score: {dummy_clf.score(X_train, y_train)}')


for i in range(1, 20):
    kneighbor_clf = KNeighborsClassifier(n_neighbors=i)
    kneighbor_clf.fit(features, target)
    print(f'Kneighbor TEST score for {i} neighbors: {kneighbor_clf.score(X_test, y_test)}')
    print(f'Kneighbor TRAIN score for {i} neighbors: {kneighbor_clf.score(X_train, y_train)}')
