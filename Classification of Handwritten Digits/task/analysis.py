# write your code here
import tensorflow as tf
import numpy as np

def set_shape(data: np.array, shape: tuple = (60000, 784)) -> np.array:
    """ set shape according to requirements"""
    data = data.reshape(shape)
    return data

def data_loader():
    # Load data for test and train
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    """
    tf.keras.datasets.mnist.load_data(path="mnist.npz") - >  
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    """

    train_n = 60000
    train_m = 28 * 28
    x_train = set_shape(x_train, (train_n, train_m))

    # reshpae:

    train_max = x_train.max()
    train_min = x_train.min()

    # print data_loader summary
    # print(f"Classes: {np.unique(y_test)}")
    # print(f"Features' shape: ({n}, {m})")
    # print(f"Target's shape: ({n},)")
    # print(f"min: {train_min}", f"max: {train_max}")
    return x_train, y_train

def train_test_hanlder(features, target):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=40, test_size=0.3)
    print(f"x_train shape:", x_train.shape)
    print(f"x_test shape:", x_test.shape)
    print(f"y_train shape:", y_train.shape)
    print(f"y_test shape:", y_test.shape)
    print("Proportion of samples per class in train set:")
    # print(x_train.value_counts(normalize=True))
    print(pd.DataFrame(y_train).value_counts(normalize=True))

if __name__ == '__main__':
    x_train, y_train = data_loader()
    x_train = x_train[0:6000, :]
    y_train = y_train[0:6000]
    train_test_hanlder(x_train, y_train)
