from model.ss_base import SemiSupervisedBase
import random
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from utils import Timer


class SemiSupervisedQBC(SemiSupervisedBase):

    def __init__(self, X, y):
        super().__init__(X, y)
        self.qbc_models = []
        self.num_committee = 3
        for i in range(self.num_committee):
            self.qbc_models.append(SGDRegressor(max_iter=500000000))

    def update_labeled(self):
        Timer.start("QBC")

        pos_list = list(range(len(self.labeled_pos_list)))
        random.shuffle(pos_list)
        i = 0
        s = int(len(self.labeled_pos_list) / self.num_committee)
        for i in range(self.num_committee):
            # Build bootstrap of training data.
            from_pos = s * i
            to_pos = s * (i + 1)
            bootstrap_labeled_pos_list = [self.labeled_pos_list[j] for j in pos_list[from_pos:to_pos]]
            # Get bootstrap training set.
            data_X_train = self.X[bootstrap_labeled_pos_list]
            # Get bootstrap target set.
            data_y_train = self.y[bootstrap_labeled_pos_list]
            # Train the model using the training sets
            self.qbc_models[i].fit(data_X_train, data_y_train)
            i += 1

        variances = []
        for pos in self.unlabeled_pos_list:
            variance = 0
            y_ave = 0
            for model in self.qbc_models:
                y = model.predict(self.X[ [pos], :])
                variance += y * y
                y_ave += y
            y_ave /= (self.num_committee * 1.0)
            variance /= (self.num_committee * 1.0)
            variance -= y_ave * y_ave
            variances.append((variance, pos))

        variances.sort(reverse = True)
        for i in range(self.batch_count):
            self.labeled_pos_list.append(variances[i][1])
            self.unlabeled_pos_list.remove(variances[i][1])
        total_time = Timer.stop("QBC")
