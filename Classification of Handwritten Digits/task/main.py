""" This scripts is used to find an optimal model and its hyper paramteres from a selection of models.
While having the flexability to support many models, it was tested on the following models:
    KNeighborsClassifier,     DecisionTreeClassifier,
    LogisticRegression ,    RandomForestClassifier.

Main Flow:
It loads training MNIST training data. Fits the models.
Evaluates the models on the test data. 
It normalizes the data (since that gave better results).
Then, it pefroms a grid search to find the optinal model

The project was based on a project in the Jetbrains Intro to Machine Learning Track.
"""

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
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

class Loader:
    """ load raw data and reshape"""
    def __init__(self):
        self.raw_data_loader()

    def raw_data_loader(self):
        # Load data for test and train
        (x_train, y_train), (x_test, y_test) = tf.kesras.datasets.mnist.load_data(path="mnist.npz")
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
        # print(f"Features' shape: ({n}, {m})")
        # print(f"Target's shape: ({n},)")
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
    def shapre_report(x_train, x_test, y_train, y_test):
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

def find_best_score(scores):
    key = max(scores.items(), key=operator.itemgetter(1))[0]
    return key, scores[key]


def format_name_and_score(best_model_name, best_score):
    model_name = f"{best_model_name}".split("(")[0] + "()"
    round_score = round(best_score, 3)
    return model_name, round_score


def remove_sister_models(scores, name1):
    for k in list(scores.keys()):
        if k.startswith(name1[:-1]):
            del scores[k]
    return scores


def print_best(scores, n=1):
    best_model_name, best_score = find_best_score(scores)
    name1, val1 = format_name_and_score(best_model_name, best_score)
    if n == 1:

        print(f"The answer to the question: {name1} - {val1}")
        return
    scores = remove_sister_models(scores, name1)
    if n == 2:
        second_best_name, second_best_score = find_best_score(scores)
        name2, val2 = format_name_and_score(second_best_name, second_best_score)
        print(f"The answer to the 2nd question: {name1}-{val1}, {name2}-{val2} ")





def run_evaluate_models(train_feats, train_target,  test_feats, test_target, print_in_run=0):
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
        if print_in_run:
            runner.print_results()
        scores.update({f"{model}": runner.score})
    return scores


def compare_score_set(scores, scores_from_normed):
    best_model, best_score = find_best_score(scores)
    best_model_normed, normed_best_score = find_best_score(scores_from_normed)
    return normed_best_score > best_score


def print_if_norm_helped(scores, scores_from_normed):
    is_b_better = compare_score_set(scores, scores_from_normed)
    if is_b_better:
       helped = "yes"
    else:
        helped = "no"
    print(f"The answer to the 1st question: {helped}\n")


def mark_scores_names_as_normed(scores_from_normed):
    refactored_score_names = {f"{key}_normed": value for key, value in scores_from_normed.items()}
    return refactored_score_names


def check_if_normed_helped_report(scores, scores_from_normed):
    print_if_norm_helped(scores, scores_from_normed)
    print_best(scores | scores_from_normed, 2)


def normalized_data(x_train, x_test):
    transformer = Normalizer().fit(x_train)  # fit does nothing.
    normed_x_train = transformer.transform(x_train)
    normed_x_test = transformer.transform(x_test)
    return normed_x_train, normed_x_test

def grid_search_random_forest(RANDOM_STATE):
    rfc_param_grid = {"n_estimators": [300, 500], "max_features": ['auto', 'log2'], "class_weight":['balanced', 'balanced_subsample']}
    rfc_gs = GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE),
                        scoring='accuracy', n_jobs=-1,param_grid=rfc_param_grid)
    return rfc_gs

def grid_search_knc(RANDOM_STATE):
    knn_param_grid = { "n_neighbors":[3, 4], "weights":['uniform', 'distance'], "algorithm":['auto', 'brute']}
    knn_gs = GridSearchCV(estimator=KNeighborsClassifier(),
                          scoring='accuracy', n_jobs=-1, param_grid=knn_param_grid)
    return knn_gs


if __name__ == '__main__':
    ##
    loader = Loader()
    splitter = Splitter()
    
    # Set shape for train data
    loader.x_train = loader.set_shape(loader.x_train)
    
    # grab only a small sample for easy manipulation
    x_train, y_train = loader.small_sample(rows=6000) # Comment this line for full sample
    
    # split data into train and test. Option - output a report on shape and values
    x_train, x_test, y_train, y_test = Splitter.train_test_hanlder(x_train, y_train)
    # Splitter.shape_report(x_train, x_test, y_train, y_test)
    
    ## Norm data
    normed_x_train, normed_x_test = normalized_data(x_train, x_test)
    
    ## Run through models - for unormalized data
    scores = run_evaluate_models(x_train, y_train, x_test, y_test) # does nothing it was only for checking
    ## Run through models - for normed data
    scores_from_normed = run_evaluate_models(normed_x_train, y_train, normed_x_test, y_test, print_in_run=0)
    
    ## compare scores
    scores_from_normed = mark_scores_names_as_normed(scores_from_normed)
    check_if_normed_helped_report(scores, scores_from_normed)
    #   Best ones are:
    #     KNeighborsClassifier
    #     RandomForestClassifier

    ## Initiate a grid search object for k-nearest neighbors and random forest
    RANDOM_STATE = 40
    knn_gs = grid_search_knc(RANDOM_STATE)
    rfc_gs = grid_search_random_forest(RANDOM_STATE)

    # fit models
    knn_gs.fit(normed_x_train, y_train)

    rfc_gs.fit(normed_x_train, y_train)

    # Show the best parameters for each model
    
    print("K-nearest neighbours algorithm")
    print(f"best estimator: {knn_gs.best_estimator_}")
    pred = knn_gs.best_estimator_.predict(normed_x_test)
    acc_score = accuracy_score(y_test, pred)
    print(f"accuracy: {round(acc_score,3)}\n")

    print("Random forest algorithm")
    print(f"best estimator: {rfc_gs.best_estimator_}")
    pred = rfc_gs.best_estimator_.predict(normed_x_test)
    print(f"accuracy: {accuracy_score(y_test, pred)}")









