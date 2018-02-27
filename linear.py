import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import math


def main():
    process("cps")



def process(name):
    with open("data/{}.dat".format(name), "rb") as infile:
        data = pickle.loads(infile.read())

    # Split the data into training/testing sets
    count = data["data"].shape[0]
    labeled_count = int(count * 0.1)
    test_count = int(count * 0.2)
    unlabeled_count = count - labeled_count - test_count
    data_X_train = data["data"][:labeled_count]
    data_X_unlabeled = data["data"][labeled_count:(labeled_count+unlabeled_count)]
    data_X_test = data["data"][(labeled_count+unlabeled_count):]

    # Split the targets into training/testing sets
    data_y_train = data["target"][:labeled_count]
    data_y_unlabeled = data["target"][labeled_count:(labeled_count+unlabeled_count)]
    data_y_test = data["target"][(labeled_count+unlabeled_count):]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(data_X_train, data_y_train)

    # Make predictions using the testing set
    data_y_pred = regr.predict(data_X_test)

    # The coefficients
    #print('Coefficients: \n', regr.coef_)

    mae = get_mean_absolute_error(data_y_test, data_y_pred)
    print("Mean absolute error: {}".format(mae))


def get_mean_absolute_error(y_actual, y_predict):
    T = y_actual.size
    mae = 0
    for i in range(T):
        mae += abs(y_actual[i] - y_predict[i])
    mae = mae / T
    return mae


if __name__ == "__main__":
    main()
