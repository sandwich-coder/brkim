#
from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np

from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, Optimizer = optim.Adam, loss_fn = nn.MSELoss()):
        if not issubclass(Optimizer, optim.Optimizer):
            raise TypeError('The optimizer should be a subclass of \'torch.nn.optim.Optimizer\'.')
        if not isinstance(loss_fn, nn.Module):
            raise TypeError('\'loss_fn\' should be a \'torch.nn.Module\'.')

        self.Optimizer = Optimizer
        self.loss_fn = loss_fn
        self.descent = None
        self.batchloss_final = None

    
    def train(self, data, model):
        if not isinstance(data, torch.Tensor):
            raise TypeError('The data should be a \'torch.Tensor\'.')
        if data.dtype != torch.float32:
            data = data.to(torch.float32)
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')

        batch_size = 32
        epochs = 10

        optimizer = self.Optimizer(
            model.parameters(),
            lr = 0.0001,
            eps = 1e-7,
            )

        loader = DataLoader(
            data,
            batch_size = batch_size,
            shuffle = True,
            )
        self.descent = []
        for lll in range(epochs):
            model.train()
            losses = []
            for x in tqdm(loader, leave = False, ncols = 70):

                out = model(x)
                loss = self.loss_fn(out, x)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.detach())    ###

            losses = torch.stack(losses, dim = 0)
            losses = losses.cpu()
            losses = losses.numpy()
            losses = losses.astype('float64')
            print('Epoch {epoch:>3} | loss: {loss_mean:<7}'.format(
                epoch = lll + 1,
                loss_mean = losses.mean(axis = 0, dtype = 'float64').round(decimals = 6),
                ))
            self.descent.append(losses)

        self.descent = np.concatenate(self.descent, axis = 0)
        self.batchloss_final = losses.mean(axis = 0, dtype = 'float64').tolist()
