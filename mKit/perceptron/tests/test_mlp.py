"""Test using MNIST dataset"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from mlp import MultiLayerPerceptron


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,
                               '{}-labels-idx1-ubyte'.format(kind))
    images_path = os.path.join(path,
                               '{}-labels-idx3-ubyte'.format(kind))

    with open(labels_path, 'rb') as lbpath:
        # Get magic number and number of items
        # Get the bytes as big endian unsigned ints
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows = struct.unpack('>IIII',
                                         imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
                             len(labels), 784)

        # Normalize the image pixel values
        images = ((images / 255.) - .5) * 2

    return images, labels


def test_mlp():
    X_train, y_train = load_mnist('data', 'train')
    # Validate row and column count on train data
    print('Rows: {}, columns: {}'.format(X_train.shape[0], X_train.shape[1]))
    if int(X_train.shape[0]) != 60000:
        print('Rows invalid -- FAIL')
    if int(X_train.shape[1]) != 784:
        print('Columns invalid -- FAIL')

    # Validate row and column count on test data
    X_test, y_test = load_mnist('data', 't10k')
    print('Rows: {}, columns: {}'.format(X_test.shape[0], X_test.shape[1]))
    if int(X_test.shape[0]) != 10000:
        print('Rows invalid -- FAIL')
    if int(X_test.shape[1]) != 784:
        print('Columns invalid -- FAIL')

    # Validate the image integrity is retained
    fig, ax = plt.subplots(nrows=2, ncols=5,
                           sharex=True, sharey=True)

    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()

    # Compress the data for portability
    np.savez_compressed('mnist_scaled.npz',
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test)

    mlp = MultiLayerPerceptron(n_hidden=100,
                               l2=0.01,
                               epochs=200,
                               eta=0.0005,
                               minibatch_size=100,
                               shuffle=True,
                               seed=1)
    # Train the network
    mlp.fit(X_train=X_train[:55000],
            y_train=y_train[:55000],
            X_valid=X_train[55000:],
            y_valid=y_train[55000:])
