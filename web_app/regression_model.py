import numpy as np


class RegressionModel():
    def __init__(self, order=1):
        self.order = order
        self.reg_poly = None

    def fit(self, X, y):
        reg_poly_coeffs = np.polyfit(X, y, self.order)
        self.reg_poly = np.poly1d(reg_poly_coeffs)

    def predict(self, X):
        return self.reg_poly(X)

    def get_reg_poly(self):
        return self.reg_poly
