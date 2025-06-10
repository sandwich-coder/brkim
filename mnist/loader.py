from copy import deepcopy as copy
import os, sys
import time
import types
import logging
import numpy as np

from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor


class Loader:
    def __init__(self):
        pass

    def load(self, name, train = True):

        array = EMNIST(
            root = 'datasets',
            split = name,
            train = train,
            download = True,
            transform = ToTensor(),
            ).data.numpy()
        array = array.astype('float64')
        array = array.reshape([array.shape[0], -1])
        array = (array - array.min()) / (array.max() - array.min())
        array = (array - np.float64(0.5)) * np.float64(2)

        return array
