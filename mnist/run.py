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
from pipe import Pipe
from autoencoder import Autoencoder
from trainer import Trainer

from sampler import Sampler
from plot import Plot


#load
loader = Loader()
array_train = loader.load('mnist')
array_test = loader.load('mnist', train = False)

#model
model = Autoencoder(Pipe())

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


#plot
plot = Plot()
np.random.seed(seed = 1)    #standardized
descent = plot.history(trainer.descent, trainer.batchloss_final)
comparisons_train = plot.before_after(
    array_train, out_train,
    index = np.random.choice(np.arange(array_train.shape[0]), size = 30, replace = False),
    )
comparisons_test = plot.before_after(
    array_test, out_test,
    index = np.random.choice(np.arange(array_test.shape[0]), size = 30, replace = False),
    )


#anomaly detection

sampler = Sampler()

anomaly = loader.load('letters', train = False)
anomaly = sampler.sample(anomaly, size = array_test.shape[0])

anomaly_re = model.flow(anomaly)

normal_error = (out_test - array_test) ** 2
normal_error = np.sqrt(normal_error.sum(axis = 1), dtype = 'float64')

anomalous_error = (anomaly_re - anomaly) ** 2
anomalous_error = np.sqrt(anomalous_error.sum(axis = 1), dtype = 'float64')

errors = plot.errors(normal_error, anomalous_error)

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
