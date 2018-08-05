import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
sys.path.append('..')
from sbs import SequentialBackwardSelection


def test_sbs():
    sc = StandardScaler()

    data = pd.read_csv('https://archive.ics.uci.edu/'
                       'ml/machine-learning-databases/'
                       'wine/wine.data', header=None)
    data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                    'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                    'Proline']
    X, y = data.iloc[:, 1].values, data.iloc[:, 0].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.3,
                         random_state=0,
                         stratify=y)

    # Standardize our inputs so KNN is not overpowered
    X_train_std = sc.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SequentialBackwardSelection(estimator=knn, k_features=1)

    sbs.fit(X_train_std, y_train)

    # Plot the accuracy of the KNN classifier
    k_features = [len(k) for k in sbs.subsets_]

    plt.plot(k_features, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()


test_sbs()
