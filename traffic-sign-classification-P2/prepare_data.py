#-*-coding=utf-8-*-
import pandas as pd
import numpy as np
import pickle

def prepare_data(training_file, validation_file, testing_file):
    training_file = training_file
    validation_file = validation_file
    testing_file = testing_file

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return (X_train, y_train, X_valid, y_valid, X_test, y_test)