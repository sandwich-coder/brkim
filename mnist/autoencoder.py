from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np

import torch
from torch import optim, nn


class Autoencoder(nn.Module):
    def __init__(self, latent):
        if not isinstance(latent, int):
            raise TypeError('\'latnet\' should be an integer.')
        if latent < 1:
            raise ValueError('\'latent\' must be positive.')
        super().__init__()
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        res = 28 * 28


        #encoder

        in_ = res
        out_ = in_ // 3 + in_ % 3
        count = 0
        while out_ > latent:
            count = count + 1
            dense = nn.ModuleDict()
            dense['affine'] = nn.Linear(in_, out_)
            dense['activation'] = nn.GELU()
            self.encoder['l'+str(count)] = dense

            in_ = out_
            out_ = in_ // 3 + in_ % 3

        dense = nn.ModuleDict()
        dense['affine'] = nn.Linear(in_, latent)
        dense['activation'] = nn.Tanh()
        self.encoder['final'] = dense


        #decoder

        in_ = latent
        out_ = in_ * 3
        count = 0
        while out_ < res:
            count = count + 1
            dense = nn.ModuleDict()
            dense['affine'] = nn.Linear(in_, out_)
            dense['activation'] = nn.GELU()
            self.decoder['l'+str(count)] = dense

            in_ = out_
            out_ = in_ * 3

        dense = nn.ModuleDict()
        dense['affine'] = nn.Linear(in_, res)
        dense['activation'] = nn.Tanh()
        self.decoder['final'] = dense


        #initialization
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder['final']['affine'].weight)
            nn.init.xavier_uniform_(self.decoder['final']['affine'].weight)


    def forward(self, x):
        x = torch.clone(x)

        for lll in self.encoder.values():
            x = lll['affine'](x)
            x = lll['activation'](x)
        
        for lll in self.decoder.values():
            x = lll['affine'](x)
            x = lll['activation'](x)

        return x
