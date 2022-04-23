# write your code here
import tensorflow as tf
import numpy as np

def set_shape(data: np.array, shape: tuple = (60000, 784)) -> np.array:
    """ set shape according to requirements"""
    data = data.reshape(shape)
    return data

def data_loader():
    # Load data for test and train
    tf.keras.datasets.mnist.load_data(path="mnist.npz")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    """
    tf.keras.datasets.mnist.load_data(path="mnist.npz") - >  
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)
    """



    train_n = 60000
    train_m = 28 * 28
    x_train = set_shape(x_train, (train_n,train_m))

    # reshpae:

    train_max = x_train.max()
    train_min = x_train.min()

    # print data_loader summary
    print(f"Classes: {np.unique(y_test)}")
    print(f"Features' shape: ({n}, {m})")
    print(f"Target's shape: ({n},)")
    print(f"min: {train_min}", f"max: {train_max}")


if __name__ == '__main__':
    data_loader()
