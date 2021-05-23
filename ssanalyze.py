import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from sklearn.linear_model import SGDRegressor
from utils import Timer
from model import SemiSupervised
from multiprocessing import Pool


class SemiSupervisedAnalyze:

    def __init__(self, name, method):
        # Initialize variables.
        self.name = name # Name of the data set to use.
        self.method = method # Name of active learning method.

        # Read data.
        with open("data/{}.pkl".format(name), "rb") as infile:
            data = pickle.loads(infile.read())

        # Instantiate one SS method
        self.ss = SemiSupervised(method, data["X"], data["y"])

    def get_average(self):
        print("Start process for {} {}...".format(self.name, self.method))
        results = self.one_run(0)
        return
        args = range(self.num_runs)
        with Pool(5) as p:
            results = p.map(self.one_run, args)

    def one_run(self, run_id):
        """
        This runs the the

        Args:
            None
        Return:
            None
        """
        Timer.start("Train {}".format(run_id))

        # Initialize
        rmse_list = []
        mae_list = []
        self.model = SGDRegressor()
        percent_labeled_list = []
        while not self.ss.is_done():
            percent_labeled, rmse, mae = self.train()
            percent_labeled_list.append(percent_labeled)
            rmse_list.append(rmse)
            mae_list.append(mae)
            self.ss.update_labeled()
        total_time = Timer.stop("Train {}".format(run_id)   )
        print("Full Training Cycle {:.2f}s".format(total_time))
        return (percent_labeled_list, rmse_list, mae_list)

    def train(self):
        # Get train and test data
        data_X_train, data_y_train = self.ss.get_labeled()
        data_X_test, data_y_test = self.ss.get_test()

        # Train the model using the training sets
        self.model.fit(data_X_train, data_y_train)

        # Make predictions using the testing set
        data_y_pred = self.model.predict(data_X_test)

        # Calculate errors
        percent_labeled = self.ss.get_percent_labeled()
        rmse = self.get_root_mean_squared(data_y_test, data_y_pred)
        mae = self.get_mean_absolute_error(data_y_test, data_y_pred)
        return (percent_labeled, rmse, mae)

    def get_mean_absolute_error(self, y_actual, y_predict):
        return np.mean(np.abs(y_actual - y_predict))

    def get_root_mean_squared(self, y_actual, y_predict):
        return np.sqrt(np.mean((y_actual - y_predict)**2))
