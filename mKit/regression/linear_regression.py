import numpy as np
from base import DataModel


class LinearRegression(DataModel):

    def __init__(self, X=[], y=[], iterations=50, alpha=0.01):
        self.X = X
        self.y = y
        self.iterations = iterations  # The number of iterations
        self.alpha = alpha  # Our learning rate
        """
        This is our feature matrix for Linear Regression
        which we will use to store the data from files
        etc that we pass through
        """
        self.features = []

    def cost_function(self, X, y, theta):
        """
        Our cost function determines how accurate
        a prediction from our hypothesis function was
        and returns it in a value, J.
        """
        m = len(y)
        J = 0

        # Convert to numpy arrays for linear algebra
        np_x = np.array(X)
        np_y = np.array(y)
        np_theta = np.array(theta)

        hypothesis = np.dot(np_theta, np_x.T)
        squared_error = (hypothesis - np_y)**2
        J = (1 / (2 * m)) * sum(squared_error)

        return J

    def gradient_descent(self, X, y, theta, alpha, num_iters=10):
        """
        The gradient descent algorithm learns
        the optimum value of theta which minimizes
        the cost function
        """
        prev_costs = []
        m = len(y)

        np_x = np.array(X)
        np_y = np.array(y)
        np_theta = np.array(theta)
        print('--------------------------------------------------------------')
        print(np_x)
        print(np_y)
        print(np_theta)
        print('--------------------------------------------------------------')

        for i in range(num_iters):
            error = np.dot(np_x, np_theta)
            delta = (1/m) * np.dot(np.subtract(error, np_y), np_x)

            theta = np_theta - alpha * delta

        return theta, prev_costs

    def feature_normalize(self, X):
        """
        Normalize the features in X where the mean
        value of each feature is 0 and the standard
        deviation is 1.
        """
        np_x = np.array(X)
        X_norm = X
        mu = np.mean(np_x)
        std = np.std(np_x)

        for i in range(1, X):
            X_norm[i] = np_x[i] - mu(i) / std(i)
        return X_norm, mu, std

    def normal_equation(self, X, y):
        """
        The normal equation allows you to calculate
        the optimum value of theta in a single go
        without needing to use gradient descent.
        """
        np_x = np.array(X)
        np_y = np.array(y)

        theta = np.linalg.inv((np_x * np_x.T)) * (np_x * np_y)

        return theta
