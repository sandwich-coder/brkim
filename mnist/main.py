import sys, os, subprocess
if sys.version_info[:2] != (3, 12):
    raise RuntimeError('This module is intended to be run on Python 3.12.')
else:
    print('Python version checked')

from basic import *
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = __name__)

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


# - plot -

os.makedirs('figures', exist_ok = True)
plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

normal = array_train.copy()
normal = sampler.sample(normal, size = 30000)
anomalous = loader.load('cloths')
anomalous = sampler.sample(anomalous, size = 30000)

#gradient descent
descent = plot.history(trainer)
descent.savefig('figures/history.png', dpi = 300)

#normal reconstructions
os.makedirs('figures/before-after-normal', exist_ok = True)
temp = np.random.choice(np.arange(len(normal)), size = 30, replace = False)
normal_reconstructions = plot.before_after(
    normal,
    temp,
    model,
    )
for l in range(len(normal_reconstructions)):
    normal_reconstructions[l].savefig('figures/before-after-normal/{index}.png'.format(
        index = temp[l],
        ), dpi = 300)

#anomaly reconstructions
os.makedirs('figures/before-after-anomalous', exist_ok = True)
temp = np.random.choice(np.arange(len(anomalous)), size = 30, replace = False)
anomalous_reconstructions = plot.before_after(
    anomalous,
    temp,
    model,
    )
for l in range(len(anomalous_reconstructions)):
    anomalous_reconstructions[l].savefig('figures/before-after-anomalous/{index}.png'.format(
        index = temp[l],
        ), dpi = 300)

#reconstruction errors
errors, error_metric = plot.errors(normal, anomalous, model, return_metric = True)
errors.savefig('figures/errors.png', dpi = 300)


# - anomaly detection (scan) -

contaminated = np.concatenate([
    sampler.sample(normal, size = 27000),
    sampler.sample(anomalous, size = 3000),
    ], axis = 0)

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

# The threshold is determined manually by observing the error plot.
errors.show()
threshold = input('threshold: ')
threshold = float(threshold)

error = error_metric(
    contaminated,
    model.flow(contaminated),
    )
prediction = np.where(error >= threshold, True, False)

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


# - anomaly detection (test) -

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

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

#Euclidean distance
error = error_metric(
    contaminated,
    model.flow(contaminated),
    )
prediction = np.where(error >= threshold, True, False)

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
