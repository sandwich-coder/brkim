from copy import deepcopy as copy
import os, sys
import time
import types
import logging
logger = logging.getLogger(name = __name__)
import numpy as np
from basic import *


class Sampler:
    def __init__(self):
        pass
    def __repr__(self):
        return 'sampler'

    def sample(self, X, size, axis = 0, replace = False):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.ndim < 1:
            raise ValueError('The input must be higher than 0-dimensional.')
        if not isinstance(size, int):
            raise TypeError('The size should be an integer.')
        if size < 1:
            raise ValueError('The size must be positive.')
        if not isinstance(axis, int):
            raise TypeError('\'axis\' should be an integer.')
        if not isinstance(replace, bool):
            raise TypeError('\'replace\' should be boolean.')
        X = X.copy()

        index = np.random.choice(
            X.shape[axis],
            size = size,
            replace = replace
            )
        sample = X.take(index, axis = axis)

        return sample
