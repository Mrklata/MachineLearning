import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


class Analytyst:
    def __init__(self):
        self.data, self.target = load_wine(return_X_y=True)
        self.kn_clf = KNeighborsClassifier(n_neighbors=3)
        self.skf = StratifiedKFold(n_splits=8)

    def score_test_stratify_random(self):
        """
        print scores
        :return: None
        """
        versions = [
            [42, None, 'No'],
            [42, self.target, 'Yes'],
            [224, self.target, 'Yes'],
            [224, None, 'No']
        ]

        for i in versions:
            x_train, x_test, y_train, y_test = train_test_split(self.data, self.target, random_state=i[0],
                                                                stratify=i[1])

            scaler = StandardScaler().fit(x_train)
            x_train_std = scaler.transform(x_train)
            x_test_std = scaler.transform(x_test)

            self.kn_clf.fit(x_train_std, y_train)

            print(f'test score for random_state={i[0]}, stratify={i[2]}: {self.kn_clf.score(x_test_std, y_test)}')

    def score_fold(self):
        """
        Print score fold data
        :return: None
        """
        mean_lst = []

        for fold, (train, test) in enumerate(self.skf.split(self.data, self.target)):
            self.kn_clf.fit(self.data[train], self.target[train])
            score = round(self.kn_clf.score(self.data[test], self.target[test]), 5)
            mean_lst.append(score)

            print(f"""
            score: {score}
            fold: {fold}
            train_bin: {np.bincount(self.target[train])}
            test_bin: {np.bincount(self.target[test])}
            train_shape: {np.shape(self.target[train])}
            test_shape: {np.shape(self.target[test])}
            """)


analytist = Analytyst()

analytist.score_test_stratify_random()
analytist.score_fold()
