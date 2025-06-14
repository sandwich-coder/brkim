from copy import deepcopy as copy
import os, sys
import time
import types
import logging
logger = logging.getLogger(__name__)
import numpy as np
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

import pandas as pd
import seaborn as sb


class Plot:
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def before_after(self, X, index, model):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be of the standard shape.')
        if not isinstance(index, np.ndarray):
            raise TypeError('The indices should be a \'numpy.ndarray\'.')
        if index.dtype != np.int64:
            raise TypeError('The indices must be of \'numpy.int64\'.')
        if index.ndim != 1:
            raise ValueError('The indices must be 1-dimensional.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        X = X.copy()

        before = X
        after = model.flow(X)

        figs = []
        for lll in index:
            fig = pp.figure(layout = 'constrained', figsize = (10, 5))
            gs = fig.add_gridspec(nrows = 1, ncols = 2)

            ax_1 = fig.add_subplot(gs[1-1])
            ax_1.set_box_aspect(1)
            ax_1.set_aspect(1)
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            plot_1 = ax_1.imshow(
                before[lll].reshape([28, 28]),
                cmap = 'grey',
                vmin = -1, vmax = 1,
                )

            ax_2 = fig.add_subplot(gs[2-1])
            ax_2.set_box_aspect(1)
            ax_2.set_aspect(1)
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            plot_2 = ax_2.imshow(
                after[lll].reshape([28, 28]),
                cmap = 'grey',
                vmin = -1, vmax = 1,
                )

            figs.append(fig)
        
        return figs


    def history(self, trainer):
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.1))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Descent', fontsize = 'medium')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot = ax.plot(
            np.arange(1, trainer.descent.shape[0]+1, dtype = 'int64'), trainer.descent,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.1,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = round(trainer.batchloss_final, ndigits = 4),
                )
            )
        ax.legend()

        return fig


    def errors(self, normal, anomalous, model):
        if not (isinstance(normal, np.ndarray) and isinstance(anomalous, np.ndarray)):
            raise TypeError('The inputs should be \'numpy.ndarray\'s.')
        if not (normal.dtype == np.float64 and anomalous.dtype == np.float64):
            raise TypeError('The inputs must be of \'numpy.float64\'.')
        if not (normal.ndim == 2 and anomalous.ndim == 2):
            raise ValueError('The inputs must be of the standard shapes.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        normal = normal.copy()
        anomalous = anomalous.copy()

        normal_out = model.flow(normal)
        anomalous_out = model.flow(anomalous)

        #Euclidean distance
        normal_error = (normal_out - normal) ** 2
        normal_error = np.sqrt(normal_error.sum(axis = 1), dtype = 'float64')
        anomalous_error = (anomalous_out - anomalous) ** 2
        anoamlous_error = np.sqrt(anomalous_error.sum(axis = 1), dtype = 'float64')

        fig = pp.figure(layout = 'constrained', figsize = (10, 7.1))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Reconstruction Errors', fontsize = 'medium')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot_1 = ax.plot(
            np.arange(1, normal_error.shape[0]+1, dtype = 'int64'), normal_error,
            marker = 'o', markersize = 0.5, alpha = 0.5,
            linestyle = '',
            color = 'tab:blue',
            label = 'normal',
            )
        plot_2 = ax.plot(
            np.arange(1, anomalous_error.shape[0]+1, dtype = 'int64'), anomalous_error,
            marker = 'o', markersize = 0.5, alpha = 0.5,
            linestyle = '',
            color = 'tab:red',
            label = 'anomalous',
            )
        ax.legend()

        return fig
