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
from plotter import Plotter
from tools.sampler import Sampler

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

normal = X.copy()
anomalous = sampler.sample(
    loader.load('cloths'),
    size = len(normal) // 9,
    )
contaminated = np.concatenate([
    normal,
    anomalous,
    ], axis = 0)
truth = np.zeros([len(contaminated)], dtype = 'int64')
truth[len(normal):] = 1
truth = truth.astype('bool')

normal_ = X_.copy()
anomalous_ = sampler.sample(
    loader.load('cloths', train = False),
    size = len(normal_) // 9,
    )
contaminated_ = np.concatenate([
    normal_,
    anomalous_
    ], axis = 0)
truth_ = np.zeros([len(contaminated_)], dtype = 'int64')
truth_[len(normal_):] = 1
truth_ = truth_.astype('bool')

#model
ae = Autoencoder()

#train
trainer = Trainer()
trainer.train(X, ae)


# - plots -

plotter = Plotter()

errors = plotter.errors(normal, anomalous, ae)
dashes = plotter.dashes(normal, ae)
violins = plotter.violins(normal, ae)

errors_ = plotter.errors(normal_, anomalous_, ae)
dashes_ = plotter.dashes(normal_, ae)
violins_ = plotter.violins(normal_, ae)

#saved
os.makedirs('figures', exist_ok = True)
errors.savefig('figures/errors-train.png', dpi = 300)
dashes.savefig('figures/dashes-train.png', dpi = 300)
violins.savefig('figures/violins-train.png', dpi = 300)
errors_.savefig('figures/errors-test.png', dpi = 300)
dashes_.savefig('figures/dashes-test.png', dpi = 300)
violins_.savefig('figures/violins-test.png', dpi = 300)


# - anomaly detection -

detector = AnomalyDetector()
detector.build(normal, anomalous, ae)

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
