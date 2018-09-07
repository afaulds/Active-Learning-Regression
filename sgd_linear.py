import numpy as np
from sklearn.linear_model import SGDRegressor


class SGDLinear:

    def __init__(self):
        self.model = SGDRegressor(loss="squared_loss", penalty="none", alpha=0.05, tol=1e-2, learning_rate="constant", warm_start=True)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, X):
        y = self.model.predict(X)
        y = np.transpose(np.asmatrix(y))
        return y
