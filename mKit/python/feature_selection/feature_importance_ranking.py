import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


class FeatureImportanceRanking(object):
    """
    Using a particular tree classifier we can use its output
    to rank the features in a dataset by how much they affect
    the data.
    --

    Parameters:
    df - The pandas dataframe holding the original data

    classifier - The classification algorithm used for determining
    the values (default random forest classifier)

    test_size - The size of the test data in the train test split
    (default 25% of the data)

    random_state - The seed value for reproducable output (default 1)
    """
    def __init__(self, df, classifier=RandomForestClassifier(n_estimators=500,
                 random_state=1),
                 test_size=0.25, random_state=1):
        self.classifier = classifier
        self.test_size = test_size
        self.random_state = random_state
        self.df = df

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                             test_size=self.test_size,
                             random_state=self.random_state)
        self.classifier.fit(X_train, y_train)

    def display_feature_importance(self, X, y):
        """
        Params
        --
        X - The x training data
        y - The y training data
        """

        importances = None
        try:
            importances = self.classifier.feature_importances_
        except AttributeError as e:
            print('This classifier is invalid: {}'.format(e))

        indices = np.argsort(importances)[::-1]
        labels = None
        try:
            labels = self.df.columns[1:]
        except Exception as e:
            print('This data frame does not contain class labels! error: {}'
                  .format(e))

        print('Feature importance list:')
        for f in range(X.shape[1]):
            print('{}) {} -- {}'.format(f + 1,
                                        labels[indices[f]],
                                        importances[indices[f]]))
        plt.title('Feature importances')
        plt.bar(range(X.shape[1]),
                importances[indices],
                align='center')

        plt.xticks(range(X.shape[1]),
                   labels, rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.tight_layout()
        plt.show()
