from basic import *
logger = logging.getLogger(name = __name__)

from torch.utils.data import DataLoader
from tqdm import tqdm


if torch.cuda.is_available():
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')
else:
    device = torch.device('cpu')
    logger.info('CPU is assigned to \'device\' as fallback.')

learning_rate = 0.0001
epsilon = 1e-7
batch_size = 32
epochs = 100


class Trainer:
    """
    reference = [
        'device',
        'learning_rate',
        'epsilon',
        'batch_size',
        'epochs',
        ]
    """
    def __init__(self, Optimizer = optim.AdamW, LossFn = nn.MSELoss):
        if not issubclass(Optimizer, optim.Optimizer):
            raise TypeError('The optimizer should be a subclass of \'torch.nn.optim.Optimizer\'.')
        if not issubclass(LossFn, nn.Module):
            raise TypeError('\'LossFn\' should be a subclass of \'torch.nn.Module\'.')

        self.Optimizer = Optimizer
        self.loss_fn = LossFn()

        self.batchloss = None
        self.trained_array = None
        self.trained_model = None

    def __repr__(self):
        return 'trainer'
    
    def train(self, X, model):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        X = X.copy()

        #processed
        data = model.process(X)

        #to gpu
        data = data.to(device)
        model.to(device)
        logger.info('\'device\' is allocated to \'data\' and \'model\'.')

        optimizer = self.Optimizer(
            model.parameters(),
            lr = learning_rate,
            eps = epsilon,
            )

        loader = DataLoader(
            data,
            batch_size = batch_size,
            shuffle = True,
            )
        self.batchloss = []
        logger.info('Training begins.')
        for lll in range(epochs):
            model.train()
            last_epoch = []
            for t in tqdm(loader, leave = False, ncols = 70):

                output = model(t)
                loss = self.loss_fn(output, t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                last_epoch.append(loss.detach())    ###


            # Even without the epoch logging, retrieving and conversion of the losses to arrays every epoch is needed to prevent the Tensors' RAM explosion.
            last_epoch = torch.stack(last_epoch, dim = 0)
            last_epoch = last_epoch.cpu()
            last_epoch = last_epoch.numpy()
            last_epoch = last_epoch.astype('float64')

            #epoch logging
            print('Epoch {epoch:>3} | loss: {epochloss:<7}'.format(
                epoch = lll + 1,
                epochloss = last_epoch.mean(axis = 0, dtype = 'float64').round(decimals = 6),
                ))

            self.batchloss.append(last_epoch)

        self.batchloss = np.concatenate(self.batchloss, axis = 0)
        self.trained_array = X.copy()
        self.trained_model = model
        logger.info(' - Training finished - ')

        #back to cpu
        model.cpu()


    def plot_losses(self):
        if self.batchloss is None:
            raise NotImplementedError('No training has been done.')
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Losses')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot = ax.plot(
            np.arange(1, len(self.batchloss)+1, dtype = 'int64'), self.batchloss,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.1,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = self.batchloss[-1].round(decimals = 4).tolist(),
                ),
            )
        ax.legend()

        return fig
