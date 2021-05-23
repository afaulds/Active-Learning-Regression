from abc import ABC, abstractmethod
import math
import random
from utils import Config


class SemiSupervisedBase:

    def __init__(self, X, y):
        # Configuration variables.
        self.max_percent = Config.get()["max_percent"] # Number of active learning loops.
        self.label_percent = Config.get()["label_percent"] # Percent of labeled data.
        self.test_percent = Config.get()["test_percent"] # Percent of test data.
        self.batch_percent = Config.get()["batch_percent"] # Percent of data to add to labeled data in each loop.

        # Set variables
        self.X = X
        self.y = y

        # Get counts
        count = self.X.shape[0]
        labeled_count = int(math.ceil(count * self.label_percent))
        test_count = int(math.ceil(count * self.test_percent))
        unlabeled_count = count - labeled_count - test_count
        self.batch_count = int(math.ceil(count * self.batch_percent))

        # Split the data into training/testing sets
        pos_list = list(range(count))
        random.shuffle(pos_list)
        self.labeled_pos_list = pos_list[:labeled_count]
        self.unlabeled_pos_list = pos_list[labeled_count:(labeled_count+unlabeled_count)]
        self.test_pos_list = pos_list[(labeled_count+unlabeled_count):]

    def is_done(self):
        return self.get_percent_labeled() > self.max_percent

    def get_percent_labeled(self):
        return 1.0 * len(self.labeled_pos_list) / self.X.shape[0]

    def get_labeled(self):
        return (
            self.X[self.labeled_pos_list],
            self.y[self.labeled_pos_list]
        )

    def get_test(self):
        return (
            self.X[self.test_pos_list],
            self.y[self.test_pos_list]
        )

    @abstractmethod
    def update_labeled(self):
        pass
