import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import math
import random
from timer import Timer


class SemiSupervisedBase:

    def __init__(self, name, method = "random"):
        self.name = name
        self.method = method
        with open("data/{}.dat".format(name), "rb") as infile:
            self.data = pickle.loads(infile.read())

    def get_average(self):
        mae = []
        for i in range(1):
            mae.append(self.process())
        mae = np.array(mae)

        # Calculate Average
        N = mae.shape[0]
        M = mae.shape[1]
        x = list(range(M))
        y_average = np.zeros(M)
        for i in range(N):
            y_average += mae[i]
        y_average /= N

        # Calculate Standard Deviation
        y_stddev = np.zeros(M)
        for i in range(N):
            y_stddev += (mae[i] - y_average) * (mae[i] - y_average)
        y_stddev = np.sqrt(y_stddev / N)

        # Build 1 stddev.
        y_top = y_average + 2 * y_stddev
        y_bottom = y_average - 2 * y_stddev

        # Plot range
        fig, ax = plt.subplots()
        ax.plot(x, y_average, x, y_top, x, y_bottom, color="black")
        ax.fill_between(x, y_average, y_top, where=y_top>y_average, facecolor="green", alpha=0.5)
        ax.fill_between(x, y_average, y_bottom, where=y_bottom<=y_average, facecolor="green", alpha=0.5)
        plt.show()


    def process(self):
        # Split the data into training/testing sets
        count = self.data["data"].shape[0]
        labeled_count = int(count * 0.1)
        test_count = int(count * 0.2)
        unlabeled_count = count - labeled_count - test_count
        self.batch_count = int(count * 0.03)
        pos_list = list(range(count))
        random.shuffle(pos_list)
        self.labeled_pos_list = pos_list[:labeled_count]
        self.unlabeled_pos_list = pos_list[labeled_count:(labeled_count+unlabeled_count)]
        self.test_pos_list = pos_list[(labeled_count+unlabeled_count):]

        mae_list = []
        for j in range(13):
            Timer.start("{} iteration".format(j))
            mae = self.train()
            mae_list.append(mae)
            self.update_labeled()
            print("Iteration {}".format(Timer.stop("{} iteration".format(j))))
        return np.array(mae_list)

    def train(self):
        data_X_train = self.data["data"][ self.labeled_pos_list ]
        data_X_test = self.data["data"][ self.test_pos_list ]

        # Split the targets into training/testing sets
        data_y_train = self.data["target"][ self.labeled_pos_list ]
        data_y_test = self.data["target"][ self.test_pos_list ]

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(data_X_train, data_y_train)

        # Make predictions using the testing set
        data_y_pred = regr.predict(data_X_test)

        # Get prediction error using mean absolute error.
        mae = get_mean_absolute_error(data_y_test, data_y_pred)
        return mae

    def update_labeled(self):
        if self.method == "random":
            self.update_labeled_random()
        elif self.method == "greedy":
            self.update_labeled_greedy()

    def update_labeled_random(self):
        Timer.start("Random")
        self.labeled_pos_list.extend(self.unlabeled_pos_list[:self.batch_count])
        self.unlabeled_pos_list = self.unlabeled_pos_list[(self.batch_count+1):]
        print("Random Update {}".format(Timer.stop("Random")))

    def update_labeled_greedy(self):
        Timer.start("Greedy")
        for i in range(self.batch_count):
            Timer.start("Single")
            max_dist = 0
            max_pos = -1
            for j in range(len(self.unlabeled_pos_list)):
                dist = self.get_min_distance(j)
                if dist > max_dist:
                    max_dist = dist
                    max_pos = self.unlabeled_pos_list[j]
            self.labeled_pos_list.append(max_pos)
            self.unlabeled_pos_list.remove(max_pos)
            print("Single Update {}".format(Timer.stop("Single")))
        print("Greedy Update {}".format(Timer.stop("Greedy")))

    def get_min_distance(self, i):
        min_dist = None
        min_pos = -1
        for j in self.labeled_pos_list:
            x = self.data["data"][i]
            y = self.data["data"][j]
            dist = np.linalg.norm(x-y)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_pos = j
        return min_dist

def get_mean_absolute_error(y_actual, y_predict):
    T = y_actual.size
    mae = 0
    for i in range(T):
        mae += abs(y_actual[i] - y_predict[i])
    mae = mae / T
    return mae
