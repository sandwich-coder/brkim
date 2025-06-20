# This is an all-in-one execution for quick experiments.

from copy import deepcopy as copy
import types
import time
import logging
logging.basicConfig(level = 'INFO')
import numpy as np
from scipy import integrate
from scipy import stats
import matplotlib as mpl
from matplotlib import pyplot as pp
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.titlesize'] = 'x-small'
mpl.rcParams['axes.labelsize'] = 'xx-small'
mpl.rcParams['xtick.labelsize'] = 'xx-small'
mpl.rcParams['ytick.labelsize'] = 'xx-small'
mpl.rcParams['legend.fontsize'] = 'x-small'
mpl.rcParams['lines.markersize'] = 1
mpl.rcParams['lines.linewidth'] = 0.5
import torch
from torch import optim, nn

from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sb
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torchvision.datasets import EMNIST, FashionMNIST
from torchvision.transforms import ToTensor


# MPS is not considered unless the 'M? Max' line.
#gpu
if torch.cuda.is_available():
    logging.info('CUDA is available.')
    device = torch.device('cuda')
    logging.info('CUDA is assigned to \'device\'.')
else:
    logging.info('GPU is not available.')
    device = torch.device('cpu')
    logging.info('CPU is assigned to \'device\' as fallback.')


# - load -

digits = {}
digits['train'] = {}
digits['test'] = {}

for l in ['train', 'test']:
    if l == 'train':
        train = True
    else:
        train = False
    
    array = MNIST(
        root = 'datasets',
        train = train,
        transform = ToTensor(),
        download = True
        ).data.numpy()
    array = array.astype('float64')
    array = array.reshape([array.shape[0], -1])
    array = (array - array.min()) / (array.max() - array.min())
    array = (array - np.float64(0.5)) * np.float64(2)
    digits[l]['original'] = array.copy()
    
del array


# - model -

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(784, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 9), nn.GELU()),
            nn.Sequential(nn.Linear(9, 5), nn.Tanh()),
            )
        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 9), nn.GELU()),
            nn.Sequential(nn.Linear(9, 27), nn.GELU()),
            nn.Sequential(nn.Linear(27, 81), nn.GELU()),
            nn.Sequential(nn.Linear(81, 243), nn.GELU()),
            nn.Sequential(nn.Linear(243, 729), nn.GELU()),
            nn.Sequential(nn.Linear(729, 784), nn.Tanh()),
            )
        
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder[-1][0].weight)
            nn.init.xavier_uniform_(self.decoder[-1][0].weight)
        
    
    def forward(self, x):
        x = torch.clone(x)
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
        


# - configure -

#optimizer and loss function
Optimizer = optim.Adam
loss_fn = nn.MSELoss()    # In machine learning, the 'loss' mostly has nothing to do with distance. It simply refers to mere difference between two scalars. Also, the 'mean' in 'Mean-Squared-Error' is not along the dimensions but along the instances. It is features, or dimensions in the spatial language, that the losses are averaged along for convenience.

#optimizer parameters
learning_rate = 0.0001
epsilon = 1e-7

#descent options
batch_size = 32
epochs = 100


# - train -

data = digits['train']['original'].copy()

scaler = MinMaxScaler(feature_range = (-1, 1))
scaler.fit(data)
data = scaler.transform(data)
data = torch.tensor(data, dtype = torch.float32)

#model
model = Autoencoder()

#gpu
data = data.to(device)
logging.info('\'device\' is allocated to \'data\'.')
model.to(device)
logging.info('\'device\' is allocated to \'model\'.')

#descent
optimizer = Optimizer(
    model.parameters(),
    lr = learning_rate,
    eps = epsilon,
    )
loader = DataLoader(data, batch_size = batch_size, shuffle = True)
descent = []
for l in range(epochs):
    model.train()
    losses = []
    for X in tqdm(loader, leave = False):
        
        out = model(X)
        loss = loss_fn(out, X)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.detach())    ###
        
    losses = torch.stack(losses, dim = 0)
    losses = losses.cpu()
    losses = losses.numpy()
    losses = losses.astype('float64')
    print('Epoch {epoch:>3} | loss: {loss_mean:<7}'.format(
        epoch = l+1,
        loss_mean = losses.mean(axis = 0, dtype = 'float64').round(decimals = 6),
        ))
    descent.append(losses)    # If 'losses' is a 'torch.Tensor' their accumulation consumes enormous RAM, for some reason I don't know. Probably some caching mechanism to optimize the training process, which becomes problematic in out-of-purpose uses.
    
descent = np.concatenate(descent, axis = 0)
batchloss_final = losses.mean(axis = 0, dtype = 'float64').round(decimals = 4)
del data, optimizer, loader, losses, X, out, loss

# The output of the 'detach' is a view that has got out of the graph. It behaves just like a "tracker", one that "follows" the change of values but not "in" the graph. The context 'torch.no_grad' is a higher abstraction of the 'detach', in whose block every operand is treated as a detached. Also, I didn't say 'computation graph' but just 'graph'. The reason is the word 'computation' is misleading. The graph in pytorch doesn't mean only "mathematical" computations, but anything a computer goes through the lines, including the shaping operations.


# - test -

model.cpu()

for l in ['train', 'test']:
    model.eval()
    
    #processed
    data = digits[l]['original'].copy()
    data = scaler.transform(data)
    data = torch.tensor(data, dtype = torch.float32)
    
    #forwarded
    output = model(data).detach()    ###
    
    #unprocessed
    _ = output.numpy()
    digits[l]['reconstructed'] = scaler.inverse_transform(_).astype('float64')
    
    #compression check
    encoded = torch.clone(data)
    encoded = model.encoder(data)
    encoded = encoded.detach()    ###
    _ = encoded.numpy()
    digits[l]['compressed'] = _.astype('float64')
    
del data, output, encoded


# - plots -

np.random.seed(seed = 1)    #standardized
cut_index = [
    100,
    101,
    598,
    1007,
    1049,
    1091,
    1101,
    1148,
    1248,
    1294,
    1758,
    1959,
    2135,
    2231,
    2239,
    3002,
    3089,
    4729,
    4749,
    5052,
    5338,
    7347,
    7353,
    8207,
    8611,
    8613,
    8617,
    8693,
    8704,
    9305,
    ]

#descent
fig = pp.figure(layout = 'constrained', figsize = (10, 7.1))
ax = fig.add_subplot()
ax.set_box_aspect(0.7)
ax.set_title('Descent', fontsize = 'medium')
ax.set_ylabel('loss')
pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')
plot = ax.plot(
    np.arange(1, descent.shape[0]+1, dtype = 'int64'), descent,
    marker = 'o', markersize = 0.3,
    linestyle = '--', linewidth = 0.1,
    color = 'slategrey',
    label = 'final: {final}'.format(
        final = batchloss_final,
        )
    )
ax.legend()
del fig, ax, plot

#before-after
index = np.random.choice(np.arange(
    digits['test']['original'].shape[0],
    ), size = 30, replace = False).tolist()
for l in index:
    before = digits['test']['original']
    after = digits['test']['reconstructed']
    fig = pp.figure(layout = 'constrained', figsize = (10, 5))
    gs = fig.add_gridspec(nrows = 1, ncols = 2)
    ax_1 = fig.add_subplot(gs[1-1])
    ax_1.set_box_aspect(1)
    ax_1.set_aspect(1)
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    plot_1 = ax_1.imshow(
        before[l].reshape([28, 28]),
        cmap = 'grey',
        vmin = -1, vmax = 1,
        )
    ax_2 = fig.add_subplot(gs[2-1])
    ax_2.set_box_aspect(1)
    ax_2.set_aspect(1)
    ax_2.set_xticks([])
    ax_2.set_yticks([])
    plot_2 = ax_2.imshow(
        after[l].reshape([28, 28]),
        cmap = 'grey',
        vmin = -1, vmax = 1,
        )
for l in cut_index:
    before = digits['train']['original']
    after = digits['train']['reconstructed']
    fig = pp.figure(layout = 'constrained', figsize = (10, 5))
    gs = fig.add_gridspec(nrows = 1, ncols = 2)
    ax_1 = fig.add_subplot(gs[1-1])
    ax_1.set_box_aspect(1)
    ax_1.set_aspect(1)
    ax_1.set_xticks([])
    ax_1.set_yticks([])
    plot_1 = ax_1.imshow(
        before[l].reshape([28, 28]),
        cmap = 'grey',
        vmin = -1, vmax = 1,
        )
    ax_2 = fig.add_subplot(gs[2-1])
    ax_2.set_box_aspect(1)
    ax_2.set_aspect(1)
    ax_2.set_xticks([])
    ax_2.set_yticks([])
    plot_2 = ax_2.imshow(
        after[l].reshape([28, 28]),
        cmap = 'grey',
        vmin = -1, vmax = 1,
        )
del index, fig, ax_1, ax_2, plot_1, plot_2

#dashes
sample = np.random.choice(np.arange(
    digits['train']['compressed'].shape[0],
    ), size = 300, replace = False)
sample = digits['train']['compressed'][sample]
fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
ax = fig.add_subplot()
ax.set_box_aspect(0.5)
ax.set_title('Dashes   (#samples: {count})'.format(
    count = sample.shape[0],
    ))
ax.set_xlabel('feature #')
ax.set_ylabel('value')
pp.setp(ax.get_yticklabels(), ha = 'right', va = 'center', rotation = 90)
plots = []
index = range(sample.shape[0])
for ll in index:
    instance = sample[ll]
    
    plot = ax.plot(
        range(1, 1+instance.shape[0]), instance,
        marker = 'o', markersize = 1.5,
        linestyle = '--', linewidth = 300 / (sample.shape[0] * sample.shape[1]),
        color = 'tab:orange',
        alpha = 0.5,
        )
    plots.append(plot)
    
ax.set_xticks(np.arange(1, 1+sample.shape[1], dtype = 'int64'))    # Plotting of the dashes forces the xticks to 'float64'.
del fig, ax, plots, plot, index

#to dataframe
feature = np.arange(1, sample.shape[1]+1, dtype = 'int64')
feature = feature.repeat(sample.shape[0], axis = 0)
value = sample.transpose().reshape([-1]).copy()
frame = np.stack([feature, value], axis = 1)
frame = pd.DataFrame(frame, columns = ['feature', 'value'])
frame['feature'] = frame['feature'].astype('int64')
del feature, value

#boxes
fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
ax = fig.add_subplot()
ax.set_box_aspect(0.5)
ax.set_title('Boxes   (#samples: {count})'.format(
    count = sample.shape[0],
    ))
ax.set_xlabel('feature #')
ax.set_ylabel('value')
pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')
sb.boxplot(
    data = frame,
    x = 'feature', y = 'value',
    orient = 'x',
    whis = (0, 100),
    color = 'tab:orange',
    ax = ax,
    )
del fig, ax

#violins
fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
ax = fig.add_subplot()
ax.set_box_aspect(0.5)
ax.set_title('Violins   (#samples: {count})'.format(
    count = sample.shape[0],
    ))
ax.set_xlabel('feature #')
ax.set_ylabel('value')
pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')
sb.violinplot(
    data = frame,
    x = 'feature', y = 'value',
    orient = 'x',
    bw_adjust = 0.5,
    inner = 'quart',
    hue = None, color = 'deepskyblue',
    density_norm = 'width',
    ax = ax,
    )
del sample, frame, fig, ax

#pp.show()
