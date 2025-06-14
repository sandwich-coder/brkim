from copy import deepcopy as copy
import os, sys
import time
import types
import logging
logger = logging.getLogger(__name__)
import numpy as np

import torch
from torch import optim, nn


class Autoencoder(nn.Module):
    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

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
