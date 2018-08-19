from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from logistic_regression import LogisticRegression


def test_logistic_regression():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=1,
                                                        stratify=y)
    X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
    y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
    lr = LogisticRegression(alpha=0.05, n_iter=1000, random_state=1)
    lr.fit(X_train_01_subset, y_train_01_subset)
    lr.plot_decision_regions(X=X_train_01_subset,
                             y=y_train_01_subset,
                             classifier=lr)
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.legend(loc='upper left')
    plt.show()


test_logistic_regression()
