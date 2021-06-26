import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

# constants
seed = np.random.seed(204)
regr = LogisticRegression()
sgd = SGDClassifier()
svc = SVC()
pipe = Pipeline([("classifier", SVC())])
skf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)

# read data
x_train = pd.read_csv("train_data.csv", header=None)
df_train_label = pd.read_csv("train_labels.csv", header=None)

label = train_label_df[0].values

x_train, x_test, y_train, y_test = train_test_split(x_train, label, test_size=0.97, shuffle=True, random_state=seed)

# scaler
scaler = StandardScaler().fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

# pca
pca = PCA(n_components=2, whiten=True, random_state=seed)
x_train_std_pca = pca.fit_transform(x_train_std)
x_test_std_pca = pca.fit_transform(x_test_std)

# setting up search area
search_area = [
    {"classifier": [regr],
     "classifier__solver": ['lbfgs', 'sag', 'saga'],
     "classifier__penalty": ['l1', 'l2', 'elasticnet', None],
     "classifier__class_weight": ["balanced", None],
     "classifier__C": np.logspace(0, 4, 10),
     "classifier__multi_class": ['ovr']},

    {"classifier": [sgd],
     "classifier__penalty": ['l1', 'l2', 'elasticnet'],
     "classifier__class_weight": [None, "balanced"],
     "classifier__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
     "classifier__loss": ['hinge', 'log', 'perceptron']},

    {"classifier": [svc],
     "classifier__kernel": ["linear", "rbf", "poly"],
     "classifier__class_weight": [None, "balanced"],
     "classifier__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
     "classifier__C": np.logspace(0, 4, 10)},
]

# grid model
grid = GridSearchCV(pipe, search_area, cv=skf, verbose=0, n_jobs=-1)

# searching for best model
best_model = gridsearch.fit(x_train_std_pca, y_train)

# best model results
print(best_model.best_estimator_.get_params()["classifier"])
print(gridsearch.best_params_)
print(gridsearch.best_score_)

# best model setp
model = LogisticRegression(C=1, class_weight=None, multi_class='ovr', penalty='l1', solver='saga')
model.fit(x_train_std_pca, y_train)

# making predictions
preds = model.predict(x_test_std_pca)

# print predictions
print(classification_report(y_test, preds))
