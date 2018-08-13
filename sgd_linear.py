import numpy as np


class SGDLinear:

    def __init__(self):
        self.alpha = 0.01
        self.num_epochs = 5
        self.coef = None
        self.inter = None
    
    def fit(self, X, y):
        # Add ones to training set.
        num_training = X.shape[0]
        xdim = X.shape[1]
        ydim = 1
        if self.coef is None:
            self.coef = np.zeros((xdim, 1))
            self.inter = np.zeros((ydim, 1))
        
        for epoch in range(self.num_epochs):
            for i in range(num_training):
                error = np.matmul(X[i, :], self.coef) + self.inter - y[i]
                self.coef = self.coef - self.alpha * np.matmul(np.transpose(X[i:(i+1), :]), error)
                self.inter = self.inter - self.alpha * error

    def predict(self, X):
        return (np.matmul(X, self.coef) + self.inter)[:, 0]
