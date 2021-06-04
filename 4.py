import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe


class Anaityst:
    def __init__(self):
        self.index, self.target = load_wine(return_X_y=True)
        self.regr = LogisticRegression(solver='liblinear')
        self.penalty = ["l1", "l2"]
        self.data = np.logspace(0, 4, 1000)
        self.h_param = dict(C=self.data, penalty=self.penalty)
        self.random_search = RandomizedSearchCV(
            self.regr,
            self.h_param,
            random_state=1,
            n_iter=1000,
            cv=5,
            verbose=0,
            n_jobs=-1
        )
        self.pipe = Pipeline([("classifier", RandomForestClassifier())])
        self.search_space = [
            {"classifier": [self.regr],
             "classifier__penalty": ["l1", "l2"],
             "classifier__C": np.logspace(0, 4, 10)},
            {"classifier": [RandomForestClassifier()],
             "classifier__n_estimators": [10, 50, 100],
             "classifier__max_features": [1, 2, 3]},
            {"classifier": [KNeighborsClassifier()],
             "classifier__n_neighbors": range(1, 10, 1),
             "classifier__leaf_size": [30, 60, 90]}
        ]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.index, self.target, test_size=0.33,
                                                                                random_state=42)

    def grid(self):
        """
        print model data
        :return: None
        """
        g_search = GridSearchCV(self.regr, self.h_param, cv=5, verbose=2, n_jobs=-1)
        model = g_search.fit(self.index, self.target)
        print(model.best_estimator_.get_params()['penalty'])
        print(model.best_estimator_.get_params()['C'])

    def random_s(self):
        """
        print random search model data
        :return: None
        """
        model = self.random_search.fit(self.index, self.target)
        print(model.best_estimator_.get_params()['penalty'])
        print(model.best_estimator_.get_params()['C'])

    def pipeline(self):
        """
        print pipeline classifier
        :return: None
        """
        grid_s = GridSearchCV(self.pipe, self.search_space, cv=5, verbose=1, n_jobs=-1)
        model = grid_s.fit(self.index, self.target)
        print(model.best_estimator_.get_params()["classifier"])

    def hyper_bot(self):
        """
        print accuracy
        :return: None
        """
        model = HyperoptEstimator(
            classifier=any_classifier("cla"),
            preprocessing=any_preprocessing("pre"),
            algo=tpe.suggest,
            max_evals=20,
            trial_timeout=30
        )
        model.fit(self.x_train, self.y_train)
        accuracy = model.score(self.x_test, self.x_train)
        print(f"Accuray: {accuracy}")


analityst = Anaityst()

analityst.grid()
analityst.random_s()
analityst.pipeline()
analityst.hyper_bot()
