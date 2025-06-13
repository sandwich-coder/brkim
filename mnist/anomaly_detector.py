from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np
import torch
from torch import optim, nn


class AnomalyDetector:
    def __init__(self, pipe, train_data, autoencoder, trainer):
        if not isinstance(train_data, torch.Tensor):
            raise TypeError('The train data should be a \'torch.Tensor\'.')
        if not isinstance(autoencoder, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')

        self.pipe = pipe
        self.train_data = train_data
        self.autoencoder = autoencoder
        self.trainer = trainer
        self.error_fn = trainer.loss_fn
        
        train_data_re = autoencoder(train_data)
        train_data_re = train_data_re.detach()    ###
        train_error = trainer.loss_fn(train_data_re, train_data)
        train_error = torch.mean(train_error, 1, dtype = torch.float32)
        train_error = train_error.numpy()
        train_error = train_error.astype('float64')
        self.threshold = np.quantile(train_error, 0.95, axis = 0)

    def __repr__(self):
        return 'anomaly detector'

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')

        data = self.pipe.process(X, train = False)
        data = torch.tensor(data, dtype = torch.float32)
        data_re = self.autoencoder(data)
        data_re = data_re.detach()    ###

        error = self.error_fn(data_re, data)
        error = torch.mean(error, 1, dtype = torch.float32)
        error = error.numpy()
        error = error.astype('float64')

        result = np.where(error >= self.threshold, True, False)
        return result
