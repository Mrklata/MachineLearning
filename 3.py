import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LassoCV, Lasso


diabetes = load_diabetes()

index, value = diabetes.data, diabetes.target
print(type(index))
kf = KFold(n_splits=10)

# split daata with kfold
for train_index, test_index in kf.split(index):
    train_x, test_x = index[train_index], index[test_index]
    train_y, test_y = value[train_index], value[test_index]
    print("TRAIN:", train_index, "\n" "TEST:", test_index)


# score for every fold
def score_by_fold():
    # lasso instance and parameters
    lasso_cv = LassoCV()
    for k, (train, test) in enumerate(kf.split(index, value)):
        lasso_cv.fit(index[train], value[train])
        print(f"fold {k+1} alpha: {lasso_cv.alpha_}, score: {lasso_cv.score(index[test], value[test])}")


def cross_validation():
    lst = [3, 5, 10]
    lasso = Lasso()
    for n in lst:
        print(cross_val_score(lasso, index, value, cv=n))
        print("Max: ", max(cross_val_score(lasso, index, value, cv=n)), "\n")


r2_score_dict = {
    # scores from 02_linear_regr
    "dummy regression": 0.0,
    "linear regression": 0.4033025232246107,
    "ridge": 0.4027277632830567,
    "lasoo": 0.40050373260020367,
    "ridgeCV": 0.4045745545779539,
    "lassoCV": 0.40050373260020367,
    # scores from 03_cross_valid
    "lasso_cv_kfold_10": 0.6827010716027995,
    "lasso_cross_val_10": 0.4287427630907267,
}
print(type(index))

r2_df = pd.DataFrame.from_dict(r2_score_dict, orient="index", columns=["value"])
sorted_r2_df = r2_df.sort_values(by=["value"])

plt.figure(figsize=(10, 5))
sns.barplot(x=r2_df.index, y=r2_df.value)
plt.ylim(0.3, 0.7)
plt.xticks(rotation=45)
plt.title("R2 scores for regression models")
plt.grid()

score_by_fold()
