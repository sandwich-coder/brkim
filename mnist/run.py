from copy import deepcopy as copy
import os, sys
import time
import types
import logging
logging.basicConfig(level = 'INFO')
import numpy as np
from scipy import integrate
from scipy import stats
from scipy.spatial.distance import pdist, cdist
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.titlesize'] = 'xx-small'
mpl.rcParams['axes.labelsize'] = 'xx-small'
mpl.rcParams['xtick.labelsize'] = 'xx-small'
mpl.rcParams['ytick.labelsize'] = 'xx-small'
mpl.rcParams['legend.fontsize'] = 'xx-small'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.linewidth'] = 0.5
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sb
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from loader import Loader
from pipe import Pipe
from autoencoder import Autoencoder
from trainer import Trainer
from plot import Plot


#gpu
if torch.cuda.is_available():
    logging.info('CUDA is available.')
    device = torch.device('cuda')
    logging.info('CUDA is assigned to \'device\'.')
else:
    logging.info('GPU is not available.')
    device = torch.device('cpu')
    logging.info('CPU is assigned to \'device\' as fallback.')

#load
loader = Loader()
data_train = loader.load('mnist')

#processed
pipe = Pipe()
data_train = pipe.process(data_train)
data_train = torch.tensor(data_train, dtype = torch.float32)

#model
model = Autoencoder()

#to gpu
data_train = data_train.to(device)
model.to(device)
logging.info('\'device\' is allocated to \'data_train\' and \'model\'.')

#train
trainer = Trainer()
trainer.train(data_train, model)

#to cpu
data_train = data_train.cpu()
model.cpu()

#test
data_test = loader.load('mnist', train = False)
data_test = torch.tensor(data_test, dtype = torch.float32)
output_train = model(data_train)
output_train = output_train.detach()
output_test = model(data_test)
output_test = output_test.detach()

#plot
plot = Plot()
np.random.seed(seed = 1)    #standardized
descent = plot.history(trainer.descent, trainer.batchloss_final)
comparisons_train = plot.before_after(
    data_train.numpy(), output_train.numpy(),
    index = np.random.choice(np.arange(data_train.shape[0]), size = 30, replace = False),
    )
comparisons_test = plot.before_after(
    data_test.numpy(), output_test.numpy(),
    index = np.random.choice(np.arange(data_test.shape[0]), size = 30, replace = False),
    )
