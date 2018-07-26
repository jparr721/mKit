import sys
sys.path.append('..')
from linear_regression import LinearRegression as linreg


def test_load_data():
    features = linreg.load(linreg, './data/test_data1.txt', ',')
    print(features)


if __name__ == '__main__':
    test_load_data()
