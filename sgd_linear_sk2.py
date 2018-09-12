import numpy as np
from sklearn.linear_model import LinearRegression


class SGDLinear:

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, train_x, train_y):
        self.model.fit(train_x, np.ravel(train_y))

    def predict(self, X):
        y = self.model.predict(X)
        y = np.transpose(np.asmatrix(y))
        return y
