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
from anomaly_detector import AnomalyDetector

from tools.sampler import Sampler
from tools.plotter import Plotter

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


#tools
sampler = Sampler()

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

#separation
normal = X.copy()
normal_ = X_.copy()
anomalous = sampler.sample(
    loader.load('cloths'),
    size = len(normal) // 11,
    )
anomalous_ = sampler.sample(
    loader.load('cloths', train = False),
    size = len(normal_) // 11,
    )


# - plot -

os.makedirs('figures', exist_ok = True)
plotter = Plotter()
np.random.seed(seed = 1)    #standardized

#normal reconstructions
os.makedirs('figures/before-after-normal', exist_ok = True)
temp = 30
if temp > len(normal):
    temp = len(normal)
temp = np.random.choice(np.arange(len(normal)), size = temp, replace = False)
normal_reconstructions = plotter.before_after(
    normal,
    model,
    index = temp,
    )
for l in range(len(normal_reconstructions)):
    normal_reconstructions[l].savefig('figures/before-after-normal/{index}.png'.format(
        index = temp[l],
        ), dpi = 300)

#dashes
dashes = plotter.dashes(normal, model, size = 500)
dashes.savefig('figures/dashes.png', dpi = 300)


# - anomaly detection -

detector = AnomalyDetector()
detector.build(normal, anomalous, model, manual = True)

#train
contaminated = np.concatenate([
    normal,
    anomalous,
    ], axis = 0)
truth = np.zeros([len(contaminated)], dtype = 'int64')
truth[len(normal):] = 1
truth = truth.astype('bool')

#test
contaminated_ = np.concatenate([
    normal_,
    anomalous_
    ], axis = 0)
truth_ = np.zeros([len(contaminated_)], dtype = 'int64')
truth_[len(normal_):] = 1
truth_ = truth_.astype('bool')

prediction = detector.predict(contaminated)
prediction_ = detector.predict(contaminated_)

#train
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

#test
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
