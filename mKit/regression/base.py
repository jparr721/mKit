import matplotlib as plt
import pandas as pd


class DataModel(object):
    def load(self, data_file):
        """
        This function reads data into
        an array to be manipulated
        """
        self.features = pd.read_csv(data_file)
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
