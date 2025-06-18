import sys, os, subprocess    # These will be imported only in the main.

from basic import *
logger = logging.getLogger(name = __name__)


class Sampler:
    def __init__(self):
        pass
    def __repr__(self):
        return 'sampler'

    def sample(self, X, size, axis = 0, replace = False):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.ndim < 1:
            raise ValueError('The array must be higher than 0-dimensional.')
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
