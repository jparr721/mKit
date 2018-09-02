import numpy as np
import matplotlib as plt
from base import DataModel


class LogisticRegression(DataModel):
    """Logistic Regression classifier using gradient descent"""
    def __init__(self, alpha=0.05, n_iter=100, random_state=1):
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the training data"""
        # Our random number generator for init values
        rgen = np.random.RandomState(self.random_state)

        # Draw initial values from the standard normal gaussian curve
        self.w_ = rgen.normal(loc=0.0, scale=0.1,
                              size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.sigmoid(net_input)
            errors = (y - output)

            self.w_[1:] += self.alpha * X.T.dot(errors)
            self.w_[0] += self.alpha * errors.sum()

            # Compute logistic cost
            cost = (-y.dot(np.log(output) - (1 - y.dot(np.log(1 - output)))))

            self.cost_.append(cost)

    def net_input(self, X):
        """Calculate the net input weights"""
        return np.dot(X, self.w_[1:]) + self.w_[0]  # w.T * x with bias term

    def sigmoid(self, z):
        """Compute the logistic regression sigmoid function"""
        return 1. / (1 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return the predicted class label after each step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


