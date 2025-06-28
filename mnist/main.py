import sys, os, subprocess

#python check
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

#gpu driver check
sh = 'nvidia-smi'
sh_ = subprocess.run('which ' + sh, shell = True, capture_output = True, text = True)
if sh_.stdout == '':
    logger.info('Command \'{command}\' does not exist.'.format(command = sh))
else:
    sh_ = subprocess.run(
        sh,
        shell = True, capture_output = True, text = True,
        )
    cuda_version = sh_.stdout.split()
    cuda_version = cuda_version[cuda_version.index('CUDA') + 2]
    if torch.version.cuda is None:
        logger.info('The installed pytorch is not built with CUDA. Install a CUDA-enabled.')
    elif float(cuda_version) < float(torch.version.cuda):
        logger.info('The supported CUDA is lower than installed. Upgrade the driver.')
    else:
        logger.info('Nvidia driver checked')


#load
loader = Loader()
X = loader.load('mnist')
X_ = loader.load('mnist', train = False)

#model
model = Autoencoder()

#train
trainer = Trainer()
trainer.train(X, model)

#test
Y = model.flow(X)
Y_ = model.flow(X_)


# - plot -

os.makedirs('figures', exist_ok = True)
plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

normal = X.copy()
anomalous = loader.load('cloths')
anomalous = sampler.sample(anomalous, size = len(normal) // 11)

#gradient descent
descent = plot.history(trainer)
descent.savefig('figures/history.png', dpi = 300)

#normal reconstructions
os.makedirs('figures/before-after-normal', exist_ok = True)
temp = 30
if temp > len(normal):
    temp = len(normal)
temp = np.random.choice(np.arange(len(normal)), size = temp, replace = False)
normal_reconstructions = plot.before_after(
    normal,
    model,
    index = temp,
    )
for l in range(len(normal_reconstructions)):
    normal_reconstructions[l].savefig('figures/before-after-normal/{index}.png'.format(
        index = temp[l],
        ), dpi = 300)

#dashes
dashes = plot.dashes(normal, model, size = 500)
dashes.savefig('figures/dashes.png', dpi = 300)

#anomaly reconstructions
os.makedirs('figures/before-after-anomalous', exist_ok = True)
temp = 30
if temp > len(anomalous):
    temp = len(anomalous)
temp = np.random.choice(np.arange(len(anomalous)), size = temp, replace = False)
anomalous_reconstructions = plot.before_after(
    anomalous,
    model,
    index = temp,
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
    normal,
    anomalous,
    ], axis = 0)

truth = np.zeros([len(contaminated)], dtype = 'int64')
truth[len(normal):] = 1
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

normal_ = X_.copy()
anomalous_ = loader.load('cloths', train = False)
anomalous_ = sampler.sample(anomalous_, size = len(normal_) // 11)

contaminated_ = np.concatenate([
    normal_,
    anomalous_
    ], axis = 0)

truth_ = np.zeros([len(contaminated_)], dtype = 'int64')
truth_[len(normal_):] = 1
truth_ = truth_.astype('bool')

#Euclidean distance
error_ = error_metric(
    contaminated_,
    model.flow(contaminated_),
    )
prediction_ = np.where(error_ >= threshold, True, False)

print('\n\n')
print('      precision (test): {precision}'.format(
    precision = precision_score(truth_, prediction_),
    ))
print('         recall (test): {recall}'.format(
    recall = recall_score(truth_, prediction_),
    ))
print('             F1 (test): {f1}'.format(
    f1 = f1_score(truth_, prediction_),
    ))
