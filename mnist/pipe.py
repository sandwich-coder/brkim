from copy import deepcopy as copy
import os, sys
import time
import types
import logging
logger = logging.getLogger(__name__)
import numpy as np
import torch
from torch import optim, nn

from sklearn.preprocessing import MinMaxScaler


class Pipe:
    def __init__(self):
        self.scaler = None
        self.previous = None
    def __repr__(self):
        return 'pipeline'

    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim != 2:
            raise ValueError('The input must be of the standard shape.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        X = X.copy()

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)
            self.scaler = scaler
            self.previous = X

        processed = self.scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)
        return processed
        

    def unprocess(self, T):
        if not isinstance(T, torch.Tensor):
            raise TypeError('The input should be a \'torch.Tensor\'.')
        if T.dtype != torch.float32:
            T = T.to(torch.float32)
        if T.dim() != 2:
            raise ValueError('The input must be of the standard shape.')
        T = torch.clone(T)

        _ = T.numpy()
        unprocessed = _.astype('float64')
        unprocessed = self.scaler.inverse_transform(unprocessed)
        return unprocessed
