import numpy as np
from sklearn.linear_model import SGDRegressor
import random


class SGDLinear:

    def __init__(self):
        self.log_transform = False
        self.model = SGDRegressor(loss="squared_loss",
            penalty="none", eta0=0.05, max_iter=1,
            learning_rate="constant", warm_start=False,
            alpha=0, tol=None)

    def fit(self, x, y):
        if self.log_transform:
            transform_y = np.log(y + 1)
        else:
            transform_y = y
        self.model.fit(x, np.ravel(transform_y))

    def predict(self, x):
        y = self.model.predict(x)
        y = np.transpose(np.asmatrix(y))
        if self.log_transform:
            y = np.exp(y) - 1
        return y
