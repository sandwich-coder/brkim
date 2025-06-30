from basic import *
logger = logging.getLogger(name = __name__)


class Sampler:
    def __init__(self):
        pass
    def __repr__(self):
        return 'sampler'

    def sample(self, A, size = None, axis = 0, replace = False):
        if not isinstance(A, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if size is not None:
            if not isinstance(size, int):
                raise TypeError('The sample size should be an integer.')
        if not isinstance(axis, int):
            raise TypeError('\'axis\' should be an integer.')
        if not isinstance(replace, bool):
            raise TypeError('\'replace\' should be boolean.')
        if not A.ndim >= 1:
            raise ValueError('The array should be at least 1-dimensional.')
        if not -A.ndim <= axis < A.ndim:
            raise ValueError('\'axis\' must be valid of the array.')
        if size is not None:
            if size <= 0:
                logger.warning('The sample size must be positive.')
                size = 1
            if size > A.shape[axis]:
                logger.warning('The sample size is bigger than the sampled.')
                size = A.shape[axis]
        else:
            if not A.shape[axis] >= 10:
                size = 1
            else:
                size = round(A.shape[axis] / 10)
        A = A.copy()

        index = np.random.choice(
            A.shape[axis],
            size = size,
            replace = replace
            )
        sample = A.take(index, axis = axis)

        return sample
