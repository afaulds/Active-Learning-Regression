from abc import ABC, abstractmethod
import math
import random


class SemiSupervisedBase:

    def __init__(self, X, y):
        # Configuration variables.
        self.num_runs = 10 # Number of runs to average for results.
        self.num_committee = 4 # Size of the committee for QBC.
        self.max_percent = 0.4 # Number of active learning loops.
        self.label_percent = 0.1 # Percent of labeled data.
        self.test_percent = 0.2 # Percent of test data.
        self.batch_percent = 0.03 #0.03 # Percent of data to add to labeled data in each loop.

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
