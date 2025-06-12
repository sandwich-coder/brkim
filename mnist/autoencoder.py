from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np

import torch
from torch import optim, nn


class Autoencoder(nn.Module):
    def __init__(self):
        if not isinstance(latent, int):
            raise TypeError('\'latnet\' should be an integer.')
        if latent < 1:
            raise ValueError('\'latent\' must be positive.')
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(784, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 9), nn.GELU()),
            nn.Sequential(nn.Linear(9, 5), nn.Tanh()),
            )

        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 9), nn.GELU()),
            nn.Sequential(nn.Linear(9, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 784), nn.Tanh()),
            )

    def forward(self, x):
        x = torch.clone(x)

        x = self.encoder(x)
        x = self.decoder(x)

        return x
