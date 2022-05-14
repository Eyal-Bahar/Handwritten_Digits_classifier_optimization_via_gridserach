# write your code here
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass, field
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import operator

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


@dataclass
class BaseRunner:
    "Base runnner without explicit implementation for a model"
    model: ...
    train_feats: ...
    train_target: ...
    test_feats: ...
    test_target: ...
    score = 0

    def __post_init__(self):
        pass

    def train(self):
        self.model.fit(self.train_feats, self.train_target)

    def predict_test(self):
        """"""
        test_predictions = self.model.predict(self.test_feats)
        return test_predictions

    def eval_(self, y_true, y_pred):
        """ accuracy from sklearn"""
        return accuracy_score(y_true, y_pred)

    def set_score(self):
        """ test predictions and compare with y_true (test_target)"""
        self.score = self.eval_(self.test_target,  self.predict_test())

    def print_results(self):
        print(f"model: {self.model}\nAccuracy: {round(self.score,3)}\n")


def fit_predict_eval(runner: BaseRunner):
    runner.fit()
    runner.set_score()
    runner.print_results()


def print_best(scores):
    key = max(scores.items(), key=operator.itemgetter(1))[0]
    model_name = f"{key}".split("(")[0] + "()"
    print(f"The answer to the question: {model_name} - {round(scores[key],3)}")



def stage_3(train_feats, train_target,  test_feats, test_target):
    RANDOM_STATE = 40
    models = [KNeighborsClassifier(),
              DecisionTreeClassifier(random_state=RANDOM_STATE),
              LogisticRegression(solver="liblinear"),
              RandomForestClassifier(random_state=RANDOM_STATE)]

    scores = {}
    for model in models:
        # model = KNeighborsClassifier() # temp
        runner = BaseRunner(model,
                            train_feats,
                            train_target,
                            test_feats,
                            test_target)
        runner.train()
        runner.set_score()
        runner.print_results()

        scores.update({f"{model}": runner.score})
    print_best(scores)

if __name__ == '__main__':
    ##
    loader = Loader()
    splitter = Splitter()
    ##from sklearn.datasets import load_iris

    loader.x_train = loader.set_shape(loader.x_train)
    x_train, y_train = loader.small_sample()
    ##
    x_train, x_test, y_train, y_test = Splitter.train_test_hanlder(x_train, y_train)
    # Splitter.stage_two_report(x_train, x_test, y_train, y_test)
    stage_3(x_train, y_train, x_test, y_test)


