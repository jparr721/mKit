import sys
sys.path.append('..')
from linear_regression import LinearRegression
import numpy as np


def test_load_data():
    lr = LinearRegression([], [], 50, 0.01)
    try:
        features = lr.load('./data/test_data1.txt')
    except Exception as e:
        print('Test failed it exception: {}'.format(e))
        return

    if features is not None:
        print('pass')
    else:
        print('fail')


def test_plot_data():
    X = [1, 2, 3, 4]
    y = [2, 3, 4, 5]
    lr = LinearRegression(X, y, 50, 0.01)
    try:
        lr.plot_data(X, y, 'X_VALS', 'Y_VALS', 'TEST_CHART')
    except Exception as e:
        print('Test failed it exception: {}'.format(e))
        return

    print('pass')


def setup():
    lr = LinearRegression([], [], 1, 1)
    features = lr.load('./data/test_data1.txt')
    return features


def run_linear_regression():
    print('Plotting data\n')
    features = setup()
    features.columns = ['Profits', 'CityPopulation']
    X = features.Profits
    y = features.CityPopulation
    m = len(y)
    iterations = 1500
    alpha = 0.01
    theta = np.zeros(m)  # Set the initial theta value
    lr = LinearRegression(X, y, iterations, alpha)

    lr.plot_data(X, y, 'Profits', 'City Population',
                 'Food Truck Profit v. City Pop')

    print('Testing gradient descent algorithm...\n')
    # Add a column of ones to X
    # X.bias = np.ones((m, 1))

    print('Initial cost: {}'.format(lr.cost_function(X, y, theta)))

    # Run the gradient descent
    theta, cost_history = lr.gradient_descent(X, y, theta, alpha, iterations)

    print('Optimum theta found by gradient descent: {}'.format(theta))
    # print('Cost history: {}'.format(cost_history))

    # Making some predictions now
    # prediction1 = np.array([1, 3.5]).T * theta
    # print(prediction1)
    # print('For population = 35,000, we predict profit of: {}'
    #       .format(prediction1))


if __name__ == '__main__':
    test_load_data()
    test_plot_data()
    run_linear_regression()
