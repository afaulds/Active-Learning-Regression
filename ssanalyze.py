import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from sklearn.linear_model import SGDRegressor
from utils import Cache, Config, Timer
from model import SemiSupervised
from multiprocessing import Pool


class SemiSupervisedAnalyze:

    def __init__(self, name, method):
        # Initialize variables.
        self.num_runs = Config.get()["num_runs"] # Number of runs to average for results.
        self.num_processors = Config.get()["num_processors"]
        self.parallel = Config.get()["use_parallel"]
        self.use_cache = Config.get()["use_cache"]
        self.name = name # Name of the data set to use.
        self.method = method # Name of active learning method.

        # Read data.
        with open("data/{}.pkl".format(name), "rb") as infile:
            self.data = pickle.loads(infile.read())

    def get_average(self):
        print("Start process for {} {}...".format(self.name, self.method))
        if self.parallel:
            args = range(self.num_runs)
            with Pool(self.num_processors) as p:
                p.map(self.run, args)
        else:
            for i in range(self.num_runs):
                self.run(i)

    def run(self, run_id):
        key = "{}_{}_{}".format(self.name, self.method, run_id)
        if not self.use_cache:
            Cache.reset()
        return Cache.process(key, self.__run, run_id)

    def __run(self, run_id):
        Timer.start("Train {}".format(run_id))

        # Initialize
        rmse_list = []
        mae_list = []
        percent_labeled_list = []

        # Instantiate classes
        ss = SemiSupervised(self.method, self.data["X"], self.data["y"])
        model = SGDRegressor(max_iter=500000000)
        while not ss.is_done():
            percent_labeled, rmse, mae = self.__train(ss, model)
            percent_labeled_list.append(percent_labeled)
            rmse_list.append(rmse)
            mae_list.append(mae)
            ss.update_labeled()
        total_time = Timer.stop("Train {}".format(run_id))
        print(rmse_list)
        print("Full Training Cycle {:.2f}s".format(total_time))
        return (percent_labeled_list, rmse_list, mae_list)

    def __train(self, ss, model):
        # Get train and test data
        data_X_train, data_y_train = ss.get_labeled()
        data_X_test, data_y_test = ss.get_test()

        # Train the model using the training sets
        model.fit(data_X_train, data_y_train)

        # Make predictions using the testing set
        data_y_pred = model.predict(data_X_test)

        # Calculate errors
        percent_labeled = ss.get_percent_labeled()
        rmse = self.get_root_mean_squared(data_y_test, data_y_pred)
        mae = self.get_mean_absolute_error(data_y_test, data_y_pred)
        return (percent_labeled, rmse, mae)

    def get_mean_absolute_error(self, y_actual, y_predict):
        return np.mean(np.abs(y_actual - y_predict))

    def get_root_mean_squared(self, y_actual, y_predict):
        return np.sqrt(np.mean((y_actual - y_predict)**2))
