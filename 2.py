import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import scatter

lasso = Lasso
ridge = Ridge
lasso_cv = LassoCV
ridge_cv = RidgeCV


def collecting_data():
    """
    :return: 4 numpy.ndarrray
    """
    diabetes = load_diabetes()

    index, values = diabetes.data, diabetes.target

    train_x, test_x, train_y, test_y = train_test_split(index, values)

    return train_x, test_x, train_y, test_y


class Base:
    def __init__(self):
        self.regression = LinearRegression()
        self.train_x, self.test_x, self.train_y, self.test_y = (
            collecting_data()[0],
            collecting_data()[1],
            collecting_data()[2],
            collecting_data()[3],
        )

    def dumny(self, strategy):
        """
        strategy{“stratified”, “most_frequent”, “prior”, “uniform”, “constant”}
        :param strategy: str - define strategy of dummy classifier
        :return: float - score of dataset
        """
        dummy_clf = DummyClassifier(strategy=strategy)
        dummy_clf.fit(self.train_x, self.train_y)
        return dummy_clf.score(self.train_x, self.train_y)

    def linear_regression(self):
        """

        :return: float - score of dataset regression
        """
        self.regression.fit(self.train_x, self.train_y)
        return self.regression.score(self.train_x, self.train_y)

    def r_rmse(self):
        """

        :return: float - score of dataset R^2 and rmse
        """
        self.regression.fit(self.train_x, self.train_y)
        prediction = self.regression.predict(self.test_x)

        return r2_score(self.test_y, prediction), mean_squared_error(
            self.test_y, prediction, squared=False
        )

    def clasiffy(self, method, alpha):
        """

        :return: 4 floats - r2 and rmse for test and train
        """
        met = method(alpha=alpha)

        met.fit(self.train_x, self.train_y)
        prediction_test = met.predict(self.test_x)
        r2_test = np.sqrt(mean_squared_error(self.test_y, prediction_test))
        rmse_test = r2_score(self.test_y, prediction_test)

        prediction_train = met.predict(self.train_x)
        r2_train = np.sqrt(mean_squared_error(self.train_y, prediction_train))
        rmse_train = r2_score(self.train_y, prediction_train)

        return r2_test, rmse_test, r2_train, rmse_train

    def cv(self, method, n_alphas, mt=None):
        """

        :return: 4 floats - r2 and rmse for test and train
        """
        if method == lasso_cv:
            mt = method(n_alphas=n_alphas, cv=4, random_state=0).fit(self.train_x, self.train_y)
        elif method == ridge_cv:
            mt = method(cv=4).fit(self.train_x, self.train_y)

        score = mt.score(self.train_x, self.train_y)

        return score


base = Base()
lasso_score_r2 = {}
ridge_score_r2 = {}

lasso_score_rmse = {}
ridge_score_rmse = {}

lasso_cv_score = {}

for i in range(1, 21):
    print(f"r_2 lasso for alpha = {i}: {base.clasiffy(lasso, i)[0]}")
    print(f"rmse lasso for alpha = {i}: {base.clasiffy(lasso, i)[1]}")

    lasso_score_r2[i] = base.clasiffy(lasso, i)[0]
    lasso_score_rmse[i] = base.clasiffy(lasso, i)[1]

    print(f"r_2 ridge for alpha = {i}: {base.clasiffy(ridge, i)[0]}")
    print(f"rmse ridge for alpha = {i}: {base.clasiffy(ridge, i)[1]}")

    ridge_score_r2[i] = base.clasiffy(ridge, i)[0]
    ridge_score_rmse[i] = base.clasiffy(ridge, i)[1]

for i in range(100, 151):

    lasso_cv_score[i] = base.cv(lasso_cv, i)


print(lasso_cv_score)


ridge_score_cv = base.cv(ridge_cv, '')

print(ridge_score_cv)


def plot():
    x, y = lasso_score_r2.keys(), lasso_score_r2.values()
    x1, y1 = ridge_score_r2.keys(), ridge_score_r2.values()

    scatter(x, y, c="#7f7f7f")
    scatter(x1, y1)

    plt.legend(["lasso", "ridge"])
    plt.show()


def plot_two():
    x2, y2 = ridge_score_rmse.keys(), ridge_score_rmse.values()
    x3, y3 = lasso_score_rmse.keys(), lasso_score_rmse.values()

    scatter(x2, y2)
    scatter(x3, y3)

    plt.legend(["lasso", "ridge"])
    plt.show()


plot()
plot_two()
