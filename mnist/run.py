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
from models import Autoencoder
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
        model.process(array_train, train = False)
        )
    encoded_test = model.encoder(
        model.process(array_test, train = False)
        )


# - plot -

plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

#gradient descent
descent = plot.history(trainer, save = True)

#before-after
comparisons_digits = plot.before_after(
    array_test,
    np.random.choice(np.arange(array_test.shape[0]), size = 30, replace = False),
    model,
    save = True,
    )

#reconstruction errors
normal = array_train.copy()
normal = sampler.sample(normal, size = 30000)
anomalous = loader.load('cloths')
anomalous = sampler.sample(anomalous, size = 30000)
errors = plot.errors(normal, anomalous, model, save = True)

#anomaly reconstructions
anomalous_reconstructions = plot.before_after(
    anomalous,
    np.random.choice(np.arange(len(anomalous)), size = 30, replace = False),
    model,
    )
os.makedirs('figures/before-after-anomalous', exist_ok = True)
for l in range(len(anomalous_reconstructions)):
    anomalous_reconstructions[l].savefig('figures/before-after-anomalous/{count}st.png'.format(
        count = l+1,
        ))

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
