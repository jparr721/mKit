import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(object):

    def __init__(self, X, y, iterations, alpha):
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

    def load(self, data_file, delimeter):
        """
        This function reads data into
        an array to be manipulated
        """
        with open(data_file) as data:
            for d in data:
                for i in d.split(delimeter):
                    self.features[i].append(i.rstrip())

        return self.features

    def plot_data(self, X, y, xlabel='X', ylabel='Y', title='X v. Y plot'):
        """
        This function plots the data points
        x and y into a new figure.
        """
        plt.plot(X, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.show()

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

        hypothesis = np_theta * np_x.T
        squared_error = (hypothesis - np_y) ^ 2
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

        for i in range(1, num_iters):
            delta = ((1 / m) * ((np_x * np_theta) - np_y) * X)

            theta = theta - alpha * delta

            prev_costs.append(self.cost_function(X, y, theta))

        return theta

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
