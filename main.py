import pandas as pd
import numpy as np
import sklearn
import os

from sklearn.datasets import load_wine
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from joblib import Memory

wine = load_wine()
features, target = wine.data, wine.target

dummy_clf = DummyClassifier()

# cache_dir = os.path.join('cache')
# memory = Memory(cache_dir)

dummy_clf.fit(features, target)
print(dummy_clf.score(features, target))


for i in range(1, 20):
    kneighbor_clf = KNeighborsClassifier(n_neighbors=i)
    kneighbor_clf.fit(features, target)
    print(kneighbor_clf.score(features, target))
