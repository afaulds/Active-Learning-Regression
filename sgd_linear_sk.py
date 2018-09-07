import numpy as np
from sklearn.linear_model import SGDRegressor


class SGDLinear:

    def __init__(self):
        self.learning_rate = 0.02
        self.num_epochs = 100
        self.model = None

    def fit(self, train_x, train_y):
        self.model = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.05, max_iter=100)
        self.model.fit(train_x, train_y)

    def predict(self, X):
        return self.model.predict(X)
