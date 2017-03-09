#!/usr/bin/python

import logging
import math
import numpy as np
import os
import random
from sklearn import datasets, linear_model
#import matplotlib.pyplot as plt
from sklearn import metrics
#from sklearn.ensemble import GradientBoostingClassifier


def main():
    raw_data = get_data("concrete.txt")
    data = build_features(raw_data, "Concretecompressivestrength")

    z = {}
    for i in range(1, 9):
        for j in range(100):
            key = i * 0.1
            idxs = setup_split(raw_data, [key, 0.2, 1])
            train_set, valid_set = split_data(data, idxs[0], idxs[1])
            model = train(train_set)
            if key not in z:
                z[key] = []
            z[key].append(validate(model, train_set, valid_set))
    for key in z:
        print("{} - {} {}".format(key, np.mean(z[key]), np.std(z[key])))

def main_old():
    #random.seed(10)

    raw_data = get_data("concrete.txt")
    idxs = setup_split(raw_data, [0.1, 0.7, 0.2])
    add_count = int(0.03 * len(raw_data)) # 3% of total

    data = build_features(raw_data, "Concretecompressivestrength")

    for i in range(23):
        train_set, valid_set = split_data(data, idxs[0], idxs[2])
        #print("")
        #print("Train Set Length : {}".format(len(train_set["data"])))
        model = train(train_set)
        validate(model, train_set, valid_set)

        idxs[0] = update_set(idxs[0], idxs[1], add_count)
        #print(train_set["data"])
        #print(train_set["target"])


def get_data(file_name):
    print("Fetching data...")
    raw_data = []
    is_first_line = True
    with open(file_name, "r") as infile:
        for line in infile:
            if is_first_line:
                keys = line.strip("\n").split("\t")
                is_first_line = False
            else:
                vals = line.strip("\n").split("\t")
                vals = dict(zip(keys, vals))
                raw_data.append(vals)
    print("Fetching data DONE.")
    return raw_data


def build_features(raw_data, target_name):
    print("Build features...")
    data = []
    target = []
    keys = list(raw_data[0].keys())
    keys.remove(target_name)
    features = keys
    for raw_item in raw_data:
        target.append(float(raw_item[target_name]))
        item = []
        for key in keys:
            item.append(float(raw_item[key]))
        data.append(item)
    print("Build features DONE.")
    return {
        "data" : np.array(data),
        "target" : np.array(target),
        "features": features
    }


def setup_split(data, ratios):
    idx = []
    for i in range(len(ratios)):
        idx.append([])
    for i in range(len(data)):
        x = random.random()
        y = 0
        for j in range(len(ratios)):
            y += ratios[j]
            if x < y:
                idx[j].append(i)
                break
    return idx


def update_set(idx, new_idx, add_count):
    x_idx = random.sample(new_idx, add_count)
    for x in x_idx:
        new_idx.remove(x)
        idx.append(x)
    return idx

def split_data(data, train_idx, valid_idx):
    train_set = {
        "data" : data["data"][train_idx],
        "target" : data["target"][train_idx],
        "features" : data["features"],
    }
    valid_set = {
        "data" : data["data"][valid_idx],
        "target" : data["target"][valid_idx],
        "features" : data["features"],
    }
    return train_set, valid_set


def train(train_set):
    data = np.empty(train_set["data"].shape)
    target = np.empty(len(train_set["target"]))
    # Calculate min and max.
    min_item = [99999999999]*len(train_set["data"][0])
    max_item = [-99999999999]*len(train_set["data"][0])
    min_target = 99999999999
    max_target = -99999999999
    for i in range(len(train_set["data"])):
        min_target = min(min_target, train_set["target"][i])
        max_target = max(max_target, train_set["target"][i])
        for j in range(len(train_set["data"][i])):
            min_item[j] = min(min_item[j], train_set["data"][i][j])
            max_item[j] = max(max_item[j], train_set["data"][i][j])
    # Convert to normalized value.
    for i in range(len(train_set["data"])):
        target[i] = (train_set["target"][i] - min_target) / (max_target - min_target)
        for j in range(len(train_set["data"][i])):
            data[i][j] = (train_set["data"][i][j] - min_item[j]) / (max_item[j] - min_item[j])
    #model = linear_model.LogisticRegression()
    model = linear_model.LinearRegression()
    model.fit(data, target)
    return {
        "model" : model,
        "min_target" : min_target,
        "max_target" : max_target,
        "min_item" : min_item,
        "max_item" : max_item,
    }


def evaluate(model, raw_data):
    data = np.empty(raw_data.shape)
    # Convert to normalized value.
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data[i][j] = (raw_data[i][j] - model["min_item"][j]) / (model["max_item"][j] - model["min_item"][j])
    yp = model["model"].predict(data)
    for i in range(len(yp)):
        yp[i] = yp[i] * (model["max_target"] - model["min_target"]) + model["min_target"]
    return yp


def validate(model, train_set, valid_set):
    #print("Features: ", train_set["features"])
    #print("Coefficients: ", model["model"].coef_)
    y = valid_set["target"]
    yp = evaluate(model, valid_set["data"])
    return rmse(y, yp)


def rmse(y, yp):
    error = 0
    for i in range(len(y)):
        error += (yp[i] - y[i]) * (yp[i] - y[i])
    return math.sqrt(error / len(y))


if __name__ == "__main__":
    main()
