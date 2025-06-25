from basic import *
logger = logging.getLogger(name = __name__)

from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(784, 1000), nn.GELU()),
            nn.Sequential(nn.Linear(1000, 333), nn.GELU()),
            nn.Sequential(nn.Linear(333, 111), nn.GELU()),
            nn.Sequential(nn.Linear(111, 37), nn.GELU()),
            nn.Sequential(nn.Linear(37, 5), nn.Tanh()),
            )

        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(5, 15), nn.GELU()),
            nn.Sequential(nn.Linear(15, 45), nn.GELU()),
            nn.Sequential(nn.Linear(45, 135), nn.GELU()),
            nn.Sequential(nn.Linear(135, 1000), nn.GELU()),
            nn.Sequential(nn.Linear(1000, 784), nn.Tanh()),
            )

        #initialized
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder[-1][0].weight)
            nn.init.xavier_uniform_(self.decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, t):
        t = torch.clone(t)

        t = self.encoder(t)
        t = self.decoder(t)

        return t


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        X = X.copy()

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)
            self.fit_scaler = scaler

        processed = self.fit_scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)
        return processed


    def unprocess(self, T):
        if not isinstance(T, torch.Tensor):
            raise TypeError('The input should be a \'torch.Tensor\'.')
        if T.dtype != torch.float32:
            T = T.to(torch.float32)
        if T.dim() != 2:
            raise ValueError('The input must be tabular.')
        T = torch.clone(T)

        _ = T.numpy()
        unprocessed = _.astype('float64')
        unprocessed = self.fit_scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        X = X.copy()

        self.eval()

        X = self.process(X, train = False)
        X = self.forward(X)
        X = X.detach()    ###
        X = self.unprocess(X)

        return X
