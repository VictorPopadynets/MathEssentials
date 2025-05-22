import numpy as np

class MyLogisticRegression:
    def __init__(self):
        self.weights = np.load("weights.npy")
        self.bias = np.load("bias.npy")

    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return (self.predict_prob(X) >= 0.5).astype(int)
