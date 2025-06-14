from basic import *
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(__name__)
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sb
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from loader import Loader
from autoencoder import Autoencoder
from trainer import Trainer

from sampler import Sampler
from plot import Plot


#load
loader = Loader()
array_train = loader.load('mnist')
array_test = loader.load('mnist', train = False)

#model
model = Autoencoder()

#train
trainer = Trainer()
trainer.train(array_train, model)

#test
out_train = model.flow(array_train)
out_test = model.flow(array_test)

#encoded
with torch.no_grad():
    encoded_train = model.encoder(
        model.pipe.process(array_train, train = False)
        )
    encoded_test = model.encoder(
        model.pipe.process(array_test, train = False)
        )


# - plot -

plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

anomaly = loader.load('letters', train = False)
anomaly = sampler.sample(anomaly, size = array_test.shape[0])

#gradient descent
descent = plot.history(trainer)

#before-after
comparisons_digits = plot.before_after(
    array_test,
    np.random.choice(np.arange(array_test.shape[0]), size = 30, replace = False),
    model,
    )
comparisons_letters = plot.before_after(
    anomaly,
    np.random.choice(np.arange(anomaly.shape[0]), size = 30, replace = False),
    model,
    )

#reconstruction errors
errors = plot.errors(array_test, anomaly, model)    # This line shows an unexpected figure. I don't know where in the plot function is wrong.

#checkpoint
pp.show()
sys.exit('\n\n--checkpoint--')


anomaly_sample = sampler.sample(anomaly, size = 1000)
contaminated = array_test.copy()
contaminated[np.random.choice(contaminated.shape[0], size = anomaly_sample.shape[0], replace = False)] = anomaly_sample

"""
print('\n\n')
print('      precision: {precision}'.format(
    precision = precision_score(truth, prediction),
    ))
print('         recall: {recall}'.format(
    recall = recall_score(truth, prediction),
    ))
print('             F1: {f1}'.format(
    f1 = f1_score(truth, prediction),
    ))
"""
