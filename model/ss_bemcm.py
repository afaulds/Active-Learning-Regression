from model.ss_base import SemiSupervisedBase
import numpy as np
import random
from sklearn.linear_model import SGDRegressor
from sklearn.utils import resample
from utils import Timer


class SemiSupervisedBEMCM(SemiSupervisedBase):

    def __init__(self, X, y):
        super().__init__(X, y)
        super().__init__(X, y)
        self.qbc_models = []
        self.num_committee = 3
        self.model = SGDRegressor(max_iter=500000000)
        for i in range(self.num_committee):
            self.qbc_models.append(SGDRegressor(max_iter=500000000))

    def update_labeled(self):
        Timer.reset("BEMCM")
        self.model.fit(self.X[self.labeled_pos_list], self.y[self.labeled_pos_list])
        for i in range(self.num_committee):
            # Build bootstrap of training data.
            bootstrap_labeled_pos_list = resample(
                self.labeled_pos_list,
                random_state=random.randrange(1000000)
            )
            data_X_train = self.X[bootstrap_labeled_pos_list]
            data_y_train = self.y[bootstrap_labeled_pos_list]

            # Train the model using the training sets
            self.qbc_models[i].fit(data_X_train, data_y_train)

        y_act = {}
        y_est = {}
        eq_24 = {}
        for pos in self.unlabeled_pos_list:
            x = self.X[ [pos] , : ]
            fx = self.model.predict(x)
            y_act[pos] = self.y[pos]
            y_est[pos] = fx
            eq_24[pos] = 0
            for j in range(len(self.qbc_models)):
                y = self.qbc_models[j].predict(x)
                eq_24[pos] += np.linalg.norm((fx - y) * x)
            eq_24[pos] /= (1.0 * len(self.qbc_models))

        for i in range(self.batch_count):
            max_change = -1
            max_pos = None
            for pos in self.unlabeled_pos_list:
                change = eq_24[pos]
                if change > max_change:
                    max_pos = pos
                    max_change = change
            del eq_24[max_pos]
            self.labeled_pos_list.append(max_pos)
            self.unlabeled_pos_list.remove(max_pos)
        Timer.stop("BEMCM")
