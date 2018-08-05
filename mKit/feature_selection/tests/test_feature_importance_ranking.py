import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('..')
from feature_importance_ranking import FeatureImportanceRanking


def test_feature_importance_ranking():
    data = pd.read_csv('https://archive.ics.uci.edu/'
                       'ml/machine-learning-databases/'
                       'wine/wine.data', header=None)
    forest = RandomForestClassifier(n_estimators=500, random_state=1)

    X, y = data.iloc[:, 1:].values, data.iloc[:, 0].values

    fir = FeatureImportanceRanking(df=data,
                                   classifier=forest,
                                   test_size=0.25,
                                   random_state=1)
    fir.fit(X, y)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=fir.test_size,
                         random_state=fir.random_state)
    fir.display_feature_importance(X_train, y_train)


test_feature_importance_ranking()
