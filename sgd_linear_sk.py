import numpy as np
from sklearn.linear_model import SGDRegressor
import random


class SGDLinear:

    def __init__(self):
        #self.model = SGDRegressor(loss="squared_loss", penalty="none", alpha=0.05, tol=1e-20, learning_rate="constant", warm_start=True, random_state=random.randrange(0,1000))
        self.model = SGDRegressor(loss="squared_loss", penalty="none", alpha=0.05, max_iter=1, random_state=random.randrange(100000))
        #self.model = SGDRegressor(loss="squared_loss", penalty="none", alpha=0.05, max_iter=1, learning_rate="constant", warm_start=True, random_state=random.randrange(0,100))

    def fit(self, train_x, train_y):
        self.model.fit(train_x, np.ravel(train_y))

    def predict(self, X):
        y = self.model.predict(X)
        y = np.transpose(np.asmatrix(y))
        return y
