from copy import deepcopy as copy
import types
import time
import logging
logger = logging.getLogger(name = __name__)
import numpy as np
from basic import *

import pandas as pd
import seaborn as sb


class Plot:
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def before_after(self, X, index, model, save = False):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be of the standard shape.')
        if not isinstance(index, np.ndarray):
            raise TypeError('The indices should be a \'numpy.ndarray\'.')
        if index.dtype != np.int64:
            raise ValueError('The indices must be of \'numpy.int64\'.')
        if index.ndim != 1:
            raise ValueError('The indices must be 1-dimensional.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(save, bool):
            raise TypeError('The \'save\' should be boolean.')
        X = X.copy()
        index = index.copy()

        before = X.copy()
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

        if save:
            os.makedirs('figures/before-after', exist_ok = True)
            for llll in range(len(index)):
                figs[llll].savefig('figures/before-after/{index}.png'.format(
                    index = index[llll],
                    ), dpi = 300)

        return figs


    def history(self, trainer, save = False):
        if not isinstance(save, bool):
            raise TypeError('The \'save\' should be boolean.')
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

        if save:
            os.makedirs('figures', exist_ok = True)
            fig.savefig('figures/history.png', dpi = 300)

        return fig


    def errors(self, normal, anomalous, model, save = False):
        if not isinstance(normal, np.ndarray):
            raise TypeError('The normal array should be a \'numpy.ndarray\'.')
        if normal.dtype != np.float64:
            normal = normal.astype('float64')
        if normal.ndim != 2:
            raise ValueError('The shape must be the dataset standard.')
        if not isinstance(anomalous, np.ndarray):
            raise TypeError('The anomalous array should be a \'numpy.ndarray\'.')
        if anomalous.dtype != np.float64:
            anomalous = anomalous.astype('float64')
        if anomalous.ndim != 2:
            raise ValueError('The shape must be the dataset standard.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(save, bool):
            raise TypeError('\'save\' should be boolean.')
        normal = normal.copy()
        anomalous = anomalous.copy()

        normal_out = model.flow(normal)
        anomalous_out = model.flow(anomalous)

        normal_error = np.sqrt(np.sum((normal_out - normal) ** 2, axis = 1), dtype = 'float64')
        anomalous_error = np.sqrt(np.sum((anomalous_out - anomalous) ** 2, axis = 1), dtype = 'float64')

        fig = pp.figure(layout = 'constrained')
        ax = fig.add_subplot()
        ax.set_box_aspect(1)
        ax.set_title('Reconstruction Errors', fontsize = 'medium')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot_1 = ax.plot(
            range(1, len(normal_error)+1), normal_error,
            marker = 'o', markersize = 8000 / len(normal_error),
            alpha = 0.8,
            linestyle = '',
            color = 'tab:blue',
            label = 'normal',
            )
        plot_2 = ax.plot(
            range(1, len(anomalous_error)+1), anomalous_error,
            marker = 'o', markersize = 8000 / len(anomalous_error),
            alpha = 0.8,
            linestyle = '',
            color = 'red',
            label = 'anomalous',
            )
        ax.axhline(
            np.median(normal_error),
            marker = '',
            linestyle = '--',
            color = 'grey',
            label = 'median (normal)',
            )
        ax.axhline(
            np.median(anomalous_error),
            marker = '',
            linestyle = '--',
            color = 'grey',
            label = 'median (anomalous)',
            )

        ax.legend()

        if save:
            os.makedirs('figures', exist_ok = True)
            fig.savefig('figures/errors.png', dpi = 300)

        return fig
