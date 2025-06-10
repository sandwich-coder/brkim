from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class Pipe:
    def __init__(self):
        self.scaler = None
        self.previous = None
    def check(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be of a standard shape.')

    def process(self, X, train = True):
        self.check(X)
        X = X.copy()

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)
            self.scaler = scaler
            self.previous = X

        processed = self.scaler.transform(X)
        return processed
        

    def unprocess(self, X):
        self.check(X)
        X = X.copy()

        unprocessed = self.scaler.inverse_transform(X)
        return unprocessed
