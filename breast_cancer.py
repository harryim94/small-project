import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

cancer = load_breast_cancer()
# print(cancer.DESCR)
# print(cancer.keys())


def create_data() -> pd.DataFrame:
    """[summary]:This function retuns data in data frame

    Returns:
        [pd.DataFrame]: returns cancer data in data frame
    """
    df = pd.DataFrame(data=cancer["data"], columns=cancer["feature_names"])
    df["target"] = cancer["target"]
    return df


# print first 5 rows of data and its shape
print(create_data().head(), create_data().shape)


def num_tar():
    """[summary]:This function print how many malignant and benign cases in the data sets
    """
    df = create_data()
    malignant = len(df[df["target"] == 0])
    benign = len(df[df["target"] == 1])
    print(pd.Series({'malignant': malignant, 'benign': benign}))
# num_tar()


def split() -> tuple:
    """[summary]:This function returns selected X,y 

    Returns:
        tuple: length of 2, pandas dataframe X and series y
    """
    df = create_data()
    X = df.iloc[:, :-1]
    y = df["target"]
    return X, y
# print(split())


def train_data() -> tuple:
    """[summary]:This function returns splited data

    Returns:
        tuple: length of 4,(X_train, X_test, y_train, y_test)
    """
    X, y = split()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


print(train_data())


def knn_train():
    """[summary]:Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).

    Returns:
        This function should return a sklearn.neighbors.classification.KNeighborsClassifier.
    """
    X_train, X_test, y_train,  y_test = train_data()
    knn = KNeighborsClassifier(n_neighbors=1)
    return knn.fit(X_train, y_train)


print(knn_train())


def knn_score() -> int:
    """[summary]:Find the score (mean accuracy) of your knn classifier using X_test and y_test.

    Returns:
        int: This function should return a float between 0 and 1
    """
    knn = knn_train()
    X_train, X_test, y_train, y_test = train_data()

    return knn.score(X_test, y_test)


print(knn_score())


def svm_train():
    """[summary]:Using Support vector machine, fit a svm classifier with X_train, y_train.

        Returns:
        This function should return a sklearn.svm.classification.LinearSVC
    """
    X_train, X_test, y_train, y_test = train_data()
    clf = LinearSVC()
    return clf.fit(X_train, y_train)


def svm_score() -> int:
    """[summary]:Find the score (accuracy) of your SVM classifier using X_test and y_test.

    Returns:
        int: This function should return a float between 0 and 1
    """
    clf = svm_train()
    X_train, X_test, y_train, y_test = train_data()

    return clf.score(X_test, y_test)


print(knn_score(), svm_score())


def best_knn():
    X_train, X_test, y_train, y_test = train_data()
    lst = range(1, 20)
    scores = []
    for n in lst:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(lst, scores)
    plt.xticks([0, 5, 10, 15, 20])
    plt.show()
