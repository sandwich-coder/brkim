from basic import *
logging.basicConfig(level = 'INFO')
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
from plot import Plot
from anomaly_detector import AnomalyDetector


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
array_train = loader.load('mnist')

#processed
pipe = Pipe()
data_train = pipe.process(array_train)
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

#processed
array_test = loader.load('mnist', train = False)
data_test = pipe.process(array_test, train = False)
data_test = torch.tensor(data_test, dtype = torch.float32)

#forwarded
with torch.no_grad():
    output_train = model(data_train)
    output_test = model(data_train)
    encoded_train = model.encoder(data_train)
    encoded_test = model.encoder(data_test)

#unprocessed
out_train = output_train.numpy()
out_train = out_train.astype('float64')
out_train = pipe.unprocess(out_train)
out_test = output_test.numpy()
out_test = out_test.astype('float64')
out_test = pipe.unprocess(out_test)


"""
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
"""


#anomaly detection

anomalous_index = np.random.choice(
    np.arange(array_test.shape[0]),
    size = 1000, replace = False,
    )
anomaly = loader.load('letters', train = False)
anomaly = anomaly[
    np.random.choice(
        np.arange(anomaly.shape[0]),
        size = anomalous_index.shape[0], replace = False,
        )
    ]

contaminated = array_test.copy()
contaminated[anomalous_index] = anomaly
detector = AnomalyDetector(pipe, data_train, model, trainer)

truth = np.zeros([array_test.shape[0]], dtype = 'bool')
truth[anomalous_index] = True
prediction = detector.predict(contaminated)

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
