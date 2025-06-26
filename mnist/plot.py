from basic import *
logger = logging.getLogger(name = __name__)

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
            raise ValueError('The array must be tabular')
        if not isinstance(index, np.ndarray):
            raise TypeError('The indices should be a \'numpy.ndarray\'.')
        if index.dtype != np.int64:
            raise ValueError('The indices must be of \'numpy.int64\'.')
        if index.ndim != 1:
            raise ValueError('The indices must be 1-dimensional.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
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

        return figs


    def history(self, trainer):
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Descent')
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


    def errors(self, normal, anomalous, model, return_metric = False):
        if not isinstance(normal, np.ndarray):
            raise TypeError('The normal should be a \'numpy.ndarray\'.')
        if normal.dtype != np.float64:
            normal = normal.astype('float64')
        if normal.ndim != 2:
            raise ValueError('The normal must be tabular.')
        if not isinstance(anomalous, np.ndarray):
            raise TypeError('The anomalous should be a \'numpy.ndarray\'.')
        if anomalous.dtype != np.float64:
            anomalous = anomalous.astype('float64')
        if anomalous.ndim != 2:
            raise ValueError('The anomalous must be tabular.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(return_metric, bool):
            raise TypeError('\'return_metric\' should be boolean.')
        normal = normal.copy()
        anomalous = anomalous.copy()

        def diff(in_, out_):
            error = (out_ - in_) ** 2
            error = error.sum(axis = 1, dtype = 'float64')
            error = np.sqrt(error, dtype = 'float64')
            return error

        normal_error = diff(
            normal,
            model.flow(normal),
            )
        anomalous_error = diff(
            anomalous,
            model.flow(anomalous),
            )

        fig = pp.figure(layout = 'constrained')
        ax = fig.add_subplot()
        ax.set_box_aspect(1)
        ax.set_title('Reconstruction Errors')
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
            linestyle = 'dashed',
            color = 'black',
            label = 'median (normal)',
            )
        ax.axhline(
            np.median(anomalous_error),
            marker = '',
            linestyle = 'dotted',
            color = 'black',
            label = 'median (anomalous)',
            )

        ax.legend()

        if return_metric:
            return fig, diff
        else:
            return fig


    #has not been tested
    def dashes(X, model, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be tabular.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if size <= 0:
            raise ValueError('\'size\' must be positive.')
        X = X.copy()

        if not sample:
            sample = X.copy()
        else:
            sample = np.random.choice(np.arange(
                len(X),
                ), size = size, replace = False)
            sample = X[sample]

        compressed = model.process(sample)
        compressed = model.encoder(compressed)
        compressed = model.unprocess(compressed)

        fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.5)
        ax.set_title('Dashes   (#samples: {count})'.format(
            count = len(compressed),
            ))
        ax.set_xlabel('feature #')
        ax.set_ylabel('value')
        pp.setp(ax.get_yticklabels(), ha = 'right', va = 'center', rotation = 90)
        plots = []
        index = range(len(compressed))
        for ll in index:
            instance = compressed[ll]

            plot = ax.plot(
                range(1, 1+len(instance)), instance,
                marker = 'o', markersize = 600 / (compressed.shape[0] * compressed.shape[1]),
                linestyle = '--', linewidth = 300 / (compressed.shape[0] * compressed.shape[1]),
                color = 'tab:orange',
                alpha = 0.5,
                )
            plots.append(plot)

        ax.set_xticks(np.arange(1, 1+compressed.shape[1], dtype = 'int64'))

        return fig
