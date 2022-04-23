# write your code here
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd

class Loader:
    """ load raw data and reshape"""
    def __init__(self):
        self.raw_data_loader()

    def raw_data_loader(self):
        # Load data for test and train
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test =  x_test, y_test


    def set_shape(self, data: np.array, shape: tuple = (60000, 784)) -> np.array:
        """ set shape according to requirements"""
        data = data.reshape(shape)
        return data

    def stage_one_report(self):
        train_max = self.x_train.max()
        train_min = self.x_train.min()

        # summary :
        print(f"Classes: {np.unique(self.y_test)}")
        print(f"Features' shape: ({n}, {m})")
        print(f"Target's shape: ({n},)")
        print(f"min: {train_min}", f"max: {train_max}")

    def small_sample(self, rows=6000):
        x_train = self.x_train[0:rows, :]
        y_train = self.y_train[0:rows]
        return x_train, y_train


class Splitter:
    def __init__(self):
        pass

    @staticmethod
    def train_test_hanlder(features, target):
        x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=40, test_size=0.3)
        return x_train, x_test, y_train, y_test
    @staticmethod
    def stage_two_report(x_train, x_test, y_train, y_test):
        print(f"x_train shape:", x_train.shape)
        print(f"x_test shape:", x_test.shape)
        print(f"y_train shape:", y_train.shape)
        print(f"y_test shape:", y_test.shape)
        print("Proportion of samples per class in train set:")
        # print(x_train.value_counts(normalize=True))
        print(pd.DataFrame(y_train).value_counts(normalize=True))


if __name__ == '__main__':
    ##
    loader = Loader()
    splitter = Splitter()
    ##
    loader.x_train = loader.set_shape(loader.x_train)
    x_train, y_train = loader.small_sample()
    ##
    x_train, x_test, y_train, y_test = Splitter.train_test_hanlder(x_train, y_train)
    Splitter.stage_two_report(x_train, x_test, y_train, y_test)
