from model.ss_base import SemiSupervisedBase
import numpy as np
from utils import Timer


class SemiSupervisedGreedy(SemiSupervisedBase):

    def __init__(self, X, y):
        super().__init__(X, y)
        self.cache = {}

    def update_labeled(self):
        Timer.reset("Greedy")
        dist_list = []
        for j in range(len(self.unlabeled_pos_list)):
            pos = self.unlabeled_pos_list[j]
            dist_list.append(self.get_min_distance(pos))

        x = sorted(zip(dist_list, self.unlabeled_pos_list), reverse=True)
        (_, pos_list) = zip(*x)

        for pos in pos_list[:self.batch_count]:
            self.labeled_pos_list.append(pos)
            self.unlabeled_pos_list.remove(pos)
        Timer.stop("Greedy")

    def get_min_distance(self, i):
        min_dist = None
        min_pos = -1
        for j in self.labeled_pos_list:
            dist = self.calc_distance(i, j)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_pos = j
        return min_dist

    def calc_distance(self, i, j):
        if i <= j:
            key = (i, j)
        else:
            key = (j, i)
        if self.cache is None:
            self.cache = {}
        if key not in self.cache:
            x = self.X[i]
            y = self.X[j]
            self.cache[key] = np.linalg.norm(x-y)
        x = self.cache[key]
        return x
