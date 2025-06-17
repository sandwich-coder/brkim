from basic import *
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(__name__)
import torch
from torch import optim, nn

import pandas as pd
import seaborn as sb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

from loader import Loader
from models.autoencoder import Autoencoder
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


# - anomaly detection -

contaminated = np.concatenate([
    sampler.sample(normal, size = 27000),
    sampler.sample(anomalous, size = 3000),
    ], axis = 0)
contaminated_out = model.flow(contaminated)

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

#Euclidean distance
error = np.sqrt(np.sum((contaminated_out - contaminated) ** 2, axis = 1), dtype = 'float64')
prediction = np.where(error >= 9, True, False)

print('\n\n')
print('     precision (train): {precision}'.format(
    precision = precision_score(truth, prediction),
    ))
print('        recall (train): {recall}'.format(
    recall = recall_score(truth, prediction),
    ))
print('            F1 (train): {f1}'.format(
    f1 = f1_score(truth, prediction),
    ))

contaminated = np.concatenate([
    sampler.sample(
        loader.load('digits', train = False),
        size = 27000,
        ),
    sampler.sample(
        loader.load('cloths', train = False),
        size = 3000,
        ),
    ], axis = 0)
contaminated_out = model.flow(contaminated)

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

#Euclidean distance
error = np.sqrt(np.sum((contaminated_out - contaminated) ** 2, axis = 1), dtype = 'float64')
prediction = np.where(error >= 9, True, False)

print('\n\n')
print('      precision (test): {precision}'.format(
    precision = precision_score(truth, prediction),
    ))
print('         recall (test): {recall}'.format(
    recall = recall_score(truth, prediction),
    ))
print('             F1 (test): {f1}'.format(
    f1 = f1_score(truth, prediction),
    ))
