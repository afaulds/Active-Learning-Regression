import numpy as np
from sklearn.linear_model import SGDRegressor
import random


class SGDLinear:

    def __init__(self):
        self.log_transform = True
        self.model = SGDRegressor(max_iter=5000)

    def fit(self, X, y):
        if self.log_transform:
            transform_y = np.log(y + 1)
        else:
            transform_y = y
        self.model.fit(X, np.ravel(transform_y))

    def predict(self, X):
        y = self.model.predict(X)
        y = np.transpose(np.asmatrix(y))
        if self.log_transform:
            y = np.exp(y) - 1
        return y
