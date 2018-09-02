import numpy as np
import sys


class MultiLayerPerceptron(object):
    """ Feedforward neural network/ Multi-layer perceptron classifier

    Parameters
    ---------
    n_hidden: int (default: 30)
        The number of hidden units

    l2: float (default: 0.)
        Lamba value for l2 regularization.
        Regularization will not happen if l2 = 0. (default)

    epochs: int (default: 100)
        Number of passes over the training set

    eta(alpha): float (default: 0.001)
        Learning rate.

    shuffle: bool (default: True)
        Shuffles the training data every epoch

    minibatch_size: int (default: 1)
        Number of training examples per minibatch

    seed: int (default: None)
        Random seed for initializing weights and shuffling
        (For reproducibility)

    Attributes
    ----------
    eval_: dict
        Dictionary containig the cost, training acccuracy,
        and validation accuracy for each epoch during training
    """
    def __init__(self, n_hidden=30, l2=0., epochs=100,
                 eta=0.001, shuffle=True,
                 minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.seed = seed

    def _onehot(self, y, n_classes):
        """One-hot encode class labels"""
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """Compute the logistic sigmoid"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """Forward propadation algorithm
        b_* values represent bias units"""
        # X is our input layer feature vector (1 x m)
        # Dot product of input layer times all hidden layers in w_h
        # z_h is our net input of the hidden layer
        z_h = np.dot(X, self.w_h) + self.b_h

        # Calcuulate the activation function of the net input layer
        a_h = self._sigmoid(z_h)

        # Calculate the net input of our single output layer
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # Use the sigmoid function to get the continuous output
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y, output):
        """Compute the cost of the neural network

        Parameters
        ----------

        y: array, shape = (n_samples, n_labels)
            one-hot encoded class labels

        output: array, shape = [n_samples, n_output_units]
            Activation of the output layer (hypothesis function)

        Returns
        -------
        cost: float
            regularized cost
        """
        # Compute the penalty (regularization parameter)
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))

        term1 = -y * (np.log(output))
        term2 = (1. - y) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """Predict the class labels

        Parameters
        ----------
        X: array, shape = [n_samples, n_features]
            Our input layer feature vector

        Returns
        -------
        y_pred: array, shape = [n_samples]
            Predicted class labels
        """
        # Forward propagate to generate values for layers
        z_h, a_h, z_out, a_out = self._forward(X)

        # Find best prediction out of the outputs
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """Backpropagate the neural network to reduce error

        Parameters
        ----------
        X_train: array, shape = [n_samples, n_features]
            Input layer feature vector from the training set

        y_train: array, shape = [n_samples]
            Taget class labels from the training set

        X_valid: array, shape = [n_samples, n_features]
            Input layer feature vector from the validation set

        y_valid: array, shape= [n_samples]
            Target class labels from the validation set


        Returns
        -------
        self
        """
        # Calculate the number of class labels
        n_output = np.unique(y_train).shape[0]

        n_features = X_train.shape[1]

        """ Weight Initialization"""

        # Weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        # Create a random normal distribution for the weight matrix
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))

        # Weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))

        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        # Encoded y_train values
        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):
            # iterate the minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                # If shuffle, shuffle the indices each iteration
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx +
                                    self.minibatch_size]

                # Forward propagate each layer
                z_h, a_h, z_out, a_out = \
                    self._forward(X_train[batch_idx])

                """Backpropagation"""

                # [n_samples, n_classlabels]
                # Compute the error at each node at each layer
                sigma_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)

                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)

                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)

                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h  # Bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h

                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # Bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            """Evaluation"""

            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y=y_train_enc, output=a_out)

            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)

            train_acc = ((np.sum(y_train ==
                          y_train_pred)).astype(np.float) /
                         X_train.shape[0])

            valid_acc = ((np.sum(y_valid ==
                          y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])

            sys.stderr.write('\n{}/{} | Cost: {}'
                             ' | Train/Valid Acc.: {}%/{}%'.format(
                                 i+1, self.epochs,
                                 cost,
                                 train_acc*100, valid_acc*100))
            sys.stderr.flush()

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)

        return self
