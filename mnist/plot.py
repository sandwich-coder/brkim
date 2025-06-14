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

import pandas as pd
import seaborn as sb


class Plot:
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def before_after(self, before, after, index):
        if not (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)):
            raise TypeError('The before and after should be a \'numpy.ndarray\'.')
        if not (before.dtype == np.float64 and after.dtype == np.float64):
            before = before.astype('float64')
            after = after.astype('float64')
        if not (before.ndim == 2 and after.ndim == 2):
            raise ValueError('The before and after must be of the dataset standard.')
        if not isinstance(index, np.ndarray):
            raise TypeError('The indices should be a \'numpy.ndarray\'.')
        if index.ndim != 1:
            raise ValueError('The indices must be 1-dimensional')

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


    def history(self, descent, batchloss_final):
        if not isinstance(descent, np.ndarray):
            raise TypeError('The descent should be a \'numpy.ndarray\'.')
        if descent.ndim != 1:
            raise ValueError('The descent must be 1-dimensional.')
        if not isinstance(batchloss_final, (int, float)):
            raise TypeError('The final batchloss should be a numebr.')
        if batchloss_final < 0:
            raise ValueError('The final batchloss must be positive.')

        fig = pp.figure(layout = 'constrained', figsize = (10, 7.1))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Descent', fontsize = 'medium')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot = ax.plot(
            np.arange(1, descent.shape[0]+1, dtype = 'int64'), descent,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.1,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = round(batchloss_final, ndigits = 4),
                )
            )
        ax.legend()

        return fig


    def errors(self, normal, anomalous):
        if not (isinstance(normal, np.ndarray) and isinstance(anomalous, np.ndarray)):
            raise TypeError('The inputs should be a \'numpy.ndarray\'.')
        if not (normal.dtype == np.float64 and anomalous.dtype == np.float64):
            normal = normal.astype('float64')
            anomalous = anomoalous.astype('float64')
        if not (normal.ndim == 1 and anomalous.ndim == 1):
            raise ValueError('The inputs must be 1-dimensional.')
        if normal.shape[0] != anomalous.shape[0]:
            raise ValueError('The inputs must have the same length.')

        fig = pp.figure(layout = 'constrained', figsize = (10, 7.1))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Reconstruction Errors', fontsize = 'medium')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot_1 = ax.plot(
            np.arange(1, normal.shape[0]+1, dtype = 'int64'), normal,
            marker = 'o', markersize = 0.5, alpha = 0.5,
            linestyle = '',
            color = 'tab:blue',
            label = 'normal',
            )
        plot_2 = ax.plot(
            np.arange(1, anomalous.shape[0]+1, dtype = 'int64'), anomalous,
            marker = 'o', markersize = 1, alpha = 0.5,
            linestyle = '',
            color = 'tab:red',
            label = 'anomalous',
            )
        ax.legend()

        return fig
