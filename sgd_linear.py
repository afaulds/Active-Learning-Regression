import math
import numpy as np


class SGDLinear:

    def __init__(self):
        self.learning_rate = 0.005
        self.num_epochs = 1
        self.coef = None
        self.inter = None

    def fit(self, x, y):
        # transform_y = np.log(y + 1)
        transform_y = y
        # Add ones to training set.
        num_training = x.shape[0]
        xdim = x.shape[1]
        ydim = 1
        if self.coef is None:
            self.coef = 0 * np.ones((xdim, 1))
            self.inter = 0 * np.ones((ydim, 1))
            return

        i_train = list(range(num_training))
        for epoch in range(self.num_epochs):
            #np.random.shuffle(i_train)
            #subset_i_train = i_train[0:1]
            subset_i_train = i_train
            for i in subset_i_train:
                error = np.matmul(x[[i], :], self.coef) + self.inter - transform_y[[i], :]
                self.coef = self.coef - self.learning_rate * np.matmul(np.transpose(x[[i], :]), error)
                self.inter = self.inter - self.learning_rate * error

    def predict(self, X):
        y = np.matmul(X, self.coef) + self.inter
        y = np.asmatrix(y)
        # y = np.exp(y) - 1
        return y
