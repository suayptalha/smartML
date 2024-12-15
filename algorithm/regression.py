import pandas as pd

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