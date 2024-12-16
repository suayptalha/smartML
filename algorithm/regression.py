import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self):
        self.m = 0
        self.b = 0
        self.r = 0

    def mean(self, values):
        return sum(values) / len(values)

    def std_dev(self, values):
        mean_val = self.mean(values)
        temp = sum((i - mean_val) ** 2 for i in values)
        temp /= len(values) - 1
        return temp ** 0.5

    def correlation_coefficient(self, x_list, x_dev, y_list, y_dev):
        x_mean = self.mean(x_list)
        y_mean = self.mean(y_list)
        temp = 0
        for i in range(len(x_list)):
            temp += ((x_list[i] - x_mean) / x_dev) * ((y_list[i] - y_mean) / y_dev)
        return temp / (len(x_list) - 1)

    def fit(self, x_list, y_list):
        x_mean = self.mean(x_list)
        y_mean = self.mean(y_list)
        x_dev = self.std_dev(x_list)
        y_dev = self.std_dev(y_list)

        self.r = self.correlation_coefficient(x_list, x_dev, y_list, y_dev)

        self.m = self.r * (y_dev / x_dev)
        self.b = y_mean - (self.m * x_mean)

    def predict(self, x):
        return self.m * x + self.b

    def equation(self):
        return f"y = {self.m:.2f}x + {self.b:.2f}"

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = self.binary_cross_entropy(y, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        y_pred = self.predict_proba(X)
        return [1 if i > 0.5 else 0 for i in y_pred]

    def equation(self):
        coef_str = " + ".join([f"{w:.2f}*x{i}" for i, w in enumerate(self.weights, start=1)])
        return f"y = sigmoid({coef_str} + {self.bias:.2f})"

class MultivariateLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = 0

    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def equation(self):
        coef_str = " + ".join([f"{w:.2f}*x{i}" for i, w in enumerate(self.weights, start=1)])
        return f"y = {coef_str} + {self.bias:.2f}"
