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
            self.last = X

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




class Autoencoder(nn.Module):
    """
    reference = [
        'Pipe',
        ]
    """
    def __init__(self):
        super().__init__()
        self.pipe = Pipe()

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(784, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 10), nn.Tanh()),
            )

        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(10, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 784), nn.Tanh()),
            )

        #initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder[-1][0].weight)
            nn.init.xavier_uniform_(self.decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'
    
    def forward(self, t):
        t = torch.clone(t)

        t = self.encoder(t)
        t = self.decoder(t)

        return t

    
    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be of the standard shape.')
        X = X.copy()

        X = self.pipe.process(X, train = False)
        X = self.forward(X)
        X = X.detach()    ###
        X = self.pipe.unprocess(X)

        return X
