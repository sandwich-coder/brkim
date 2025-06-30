from basic import *
logger = logging.getLogger(name = __name__)

import pandas as pd
import seaborn as sb

from tools.sampler import Sampler

sampler = Sampler()


class Plotter:
    """
    reference = [
        sampler,
        ]
    """
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def before_after(self, X, ae, index = None):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be tabular')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if index is not None:
            if not isinstance(index, np.ndarray):
                raise TypeError('The indices should be a \'numpy.ndarray\'.')
            if index.dtype != np.int64:
                raise ValueError('The indices must be of \'numpy.int64\'.')
            if index.ndim != 1:
                raise ValueError('The indices must be 1-dimensional.')
            index = index.copy()
        else:
            index = np.random.choice(len(X), size = 30, replace = False)
        X = X.copy()

        before = X.copy()
        after = ae.flow(X)

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


    def dashes(self, X, ae, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if not X.ndim == 2:
            raise ValueError('The array must be tabular.')
        if not size > 0:
            raise ValueError('\'size\' must be positive.')
        if not X.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        if not sample:
            sample = X.copy()
        else:
            sample = sampler.sample(X, size = size)

        compressed = ae.process(sample, train = False)
        compressed = ae.encoder(compressed)
        compressed = compressed.detach()    ###
        compressed = compressed.numpy()

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
                marker = 'o', markersize = 3 / len(compressed) ** 0.5,
                linestyle = '--', linewidth = 3 / len(compressed),
                alpha = 0.8,
                color = 'tab:orange',
                )
            plots.append(plot)

        ax.set_xticks(np.arange(1, 1+compressed.shape[1], dtype = 'int64'))

        return fig
