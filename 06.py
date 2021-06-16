import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.svm import SVC


seed = np.random.seed(204)
x, y = load_wine(return_X_y=True)

# Standard scaler
scaler = StandardScaler().fit(x)
x_standard = scaler.transform(x)

# Load CSV data
train_data = pd.read_csv("train_data.csv", header=None)
test_data = pd.read_csv("test_data.csv", header=None)
train_labels = pd.read_csv("project_data/train_labels.csv", header=None)

# Split data
train_data_02, train_data_08 = train_test_split(
    train_data, test_size=0.8, shuffle=True, random_state=seed
)

test_data_02, test_data_08 = train_test_split(
    test_data, test_size=0.8, shuffle=True, random_state=seed
)

# Standard scaler for splited data
scaler = StandardScaler().fit(test_data_02)
data_std = scaler.transform(test_data_02)

# PCA
pca = PCA(n_components=2, whiten=True, random_state=seed)
pca_data = pca.fit_transform(data_std)

# TSNE
tsne = TSNE(n_components=2, random_state=seed)
tsne_data = tsne.fit_transform(data_std)

# Data for ploting
data_sets = [pca_data, tsne_data]
names = ["pca_data", "tsne_data"]
colors = ["red", "green", "blue"]

# Pipeline
pipe = Pipeline(
    [
        ("std", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=seed)),
        ("tsne", TSNE(n_components=2, random_state=seed)),
    ]
)
piped = pipe.fit_transform(test_data_02)


def diabetes_classification() -> None:
    elements = [2, 4, 6]
    sum_of_elements = []
    for element in elements:
        pca = PCA(n_components=element, random_state=seed)
        pca.fit(x_standard)

        ratio = pca.explained_variance_ratio_
        print(ratio)

        ratio_sum = sum(ratio)
        sum_of_elements.append(ratio_sum)

    df_to_plot = pd.DataFrame(sum_of_elements, index=elements, columns=["ratio"])
    df_to_plot.index.name = "n of components"
    df_to_plot.plot(kind="bar", grid=True)


def pca_tsne() -> None:
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    print(tsne.kl_divergence_)


def plot() -> None:
    for i, n, c in zip(data_sets, names, colors):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            i[:, 0], i[:, 1], s=75, c=c, marker="o", alpha=0.5, edgecolor="black"
        )
        plt.title(n)


# fit transform method


def claster() -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(
        piped[:, 0], piped[:, 1], s=75, c="b", marker="o", alpha=0.6, edgecolor="black"
    )

    plt.title(names)
    plt.grid()
    plt.show()


def grid_search() -> None:
    y2 = train_labels[0].values

    x_train, x_test, y_train, y_test = train_test_split(
        train_data, y2, test_size=0.95, shuffle=True, random_state=seed
    )

    scaler = StandardScaler().fit(x_train)
    x_train_standard = scaler.transform(x_train)

    kpca = KernelPCA()
    svc_model = SVC()

    pipe = Pipeline(steps=[("kpca", kpca), ("svc_model", svc_model)])
    pipe.fit(x_train_standard, y_train)

    param_dict = {
        "kpca__gamma": np.linspace(0.03, 0.05, 5),
        "kpca__kernel": ["linear", "poly", "rbf"],
        "svc_model__C": [0.1, 1, 10, 100, 1000],
        "svc_model__gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    }

    grid = GridSearchCV(pipe, param_dict, verbose=0)
    grid.fit(x_train_standard, y_train)

    print(f"best param:\n{grid.best_params_}")
    print(f"best score:\n{grid.best_score_}")
