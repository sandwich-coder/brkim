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
latent = 5
model = Autoencoder(latent = latent)

#to gpu
data_train = data_train.to(device)
model.to(device)
logging.info('\'device\' is allocated to \'data_train\' and \'model\'.')

#train
trainer = Trainer()
trainer.train(data_train, model)
