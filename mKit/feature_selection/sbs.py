from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SequentialBackwardSelection(object):
    """
    Sci-Kit learn's missing SBS algorithm implementation
    ---

    Paramteters:
    estimator - The Machine Learning algorithm we will use for
    learning and estimating the provided data

    k_features - The number of features we want to return (must be
    smaller than number of provided features)

    scoring - The system to evaluate the performance of a particular
    model (default sci-kit's accuracy_score)

    test_size - The desired size for the test data when it is split

    random_state - The seed for the randomness of the produced data
    when splitting, defaults to 1 for reproducibility
    """
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size,
                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.inidicies_ = tuple(range(dim))
        self.subsets_ = [self.inidicies_]
        score = self.calc_score(X_train,
                                y_train,
                                X_test,
                                y_test,
                                self.inidicies_)

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.inidicies_, r=dim - 1):
                score = self.calc_score(X_train, y_train,
                                        X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.inidicies_ = subsets[best]
            self.subsets_.append(self.inidicies_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.inidicies_]

    def calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_test)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
