"""
SLAD: Adapted from the official implementation of "Deep anomaly detection with scale learning"
    https://github.com/xuhongzuo/scale-learning
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import time
from abc import ABCMeta, abstractmethod
import importlib


class BaseDeepAD(metaclass=ABCMeta):
    def __init__(self, model_name, epochs=100, batch_size=64, lr=1e-3,
                 epoch_steps=-1, prt_steps=10, device='cuda:0',
                 verbose=1, random_state=42):
        self.model_name = model_name

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.device = device

        self.epoch_steps = epoch_steps
        self.prt_steps = prt_steps
        self.verbose = verbose

        self.n_features = -1
        self.n_samples = -1
        self.criterion = None
        self.net = None

        self.train_loader = None
        self.test_loader = None

        self.epoch_time = None

        self.random_state=random_state
        # self.set_seed(random_state)
        return

    def fit(self, X, y=None):
        """
        Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.n_samples, self.n_features = X.shape

        self.train_loader, self.net, self.criterion = self.training_prepare(X, y)
        self.training()

        return self

    def predict_score(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        self.test_loader = self.inference_prepare(X)
        scores = self.inference()

        return scores

    def training(self):
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.lr,
                                     weight_decay=1e-5)

        self.net.train()
        for i in range(self.epochs):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in self.train_loader:
                loss = self.training_forward(batch_x, self.net, self.criterion)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break

            t = time.time() - t1
            if self.verbose >=1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1}, '
                      f'training loss: {total_loss/cnt:.6f}, '
                      f'time: {t:.1f}s')

            if i == 0:
                self.epoch_time = t

            self.epoch_update()

        return

    def inference(self):

        self.net.eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []
            for batch_x in self.test_loader:
                batch_z, s = self.inference_forward(batch_x, self.net, self.criterion)

                z_lst.append(batch_z)
                score_lst.append(s)

        z = torch.cat(z_lst).data.cpu().numpy()
        scores = torch.cat(score_lst).data.cpu().numpy()

        return scores

    @abstractmethod
    def training_forward(self, batch_x, net, criterion):
        pass

    @abstractmethod
    def inference_forward(self, batch_x, net, criterion):
        pass

    @abstractmethod
    def training_prepare(self, X, y=None):
        """define train_loader, net, and criterion"""
        pass

    @abstractmethod
    def inference_prepare(self, X):
        pass

    def epoch_update(self):
        """for any updating operation after each training epoch"""
        return
    

class MLPnet(torch.nn.Module):
    def __init__(self, n_features, n_hidden=[500, 100], n_output=20,
                 activation='ReLU', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(MLPnet, self).__init__()
        self.skip_connection = skip_connection
        self.n_output = n_output

        if type(n_hidden)==int:
            n_hidden = [n_hidden]
        if type(n_hidden)==str:
            n_hidden = n_hidden.split(',')
            n_hidden = [int(a) for a in n_hidden]

        num_layers = len(n_hidden)

        self.layers = []
        for i in range(num_layers+1):
            in_channels, out_channels = self.get_in_out_channels(i, num_layers, n_features,
                                                                 n_hidden, n_output, skip_connection)
            self.layers += [
                LinearBlock(in_channels, out_channels,
                            bias=bias, batch_norm=batch_norm,
                            activation=activation if i != num_layers else None,
                            skip_connection=skip_connection if i != num_layers else 0,
                            dropout=dropout)
            ]
        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.network(x)
        return x

    def get_in_out_channels(self, i, num_layers, n_features, n_hidden, n_output, skip_connection):
        if skip_connection is None:
            in_channels = n_features if i == 0 else n_hidden[i-1]
            out_channels = n_output if i == num_layers else n_hidden[i]
        elif skip_connection == 'concat':
            in_channels = n_features if i == 0 else np.sum(n_hidden[:i])+n_features
            out_channels = n_output if i == num_layers else n_hidden[i]
        else:
            raise NotImplementedError('')
        return in_channels, out_channels


class LinearBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation='tanh', bias=False, batch_norm=False,
                 skip_connection=None, dropout=None):
        super(LinearBlock, self).__init__()

        self.skip_connection = skip_connection

        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)

        # Tanh, ReLU, LeakyReLU, Sigmoid
        if activation is not None:
            self.act_layer = instantiate_class("torch.nn.modules.activation", activation)
        else:
            self.act_layer = torch.nn.Identity()

        self.dropout = dropout
        if dropout is not None:
            self.dropout_layer = torch.nn.Dropout(p=dropout)

        self.batch_norm = batch_norm
        if batch_norm is True:
            self.bn_layer = torch.nn.BatchNorm1d(out_channels, affine=bias)


    def forward(self, x):
        x1 = self.linear(x)
        x1 = self.act_layer(x1)

        if self.batch_norm is True:
            x1 = self.bn_layer(x1)

        if self.dropout is not None:
            x1 = self.dropout_layer(x1)

        if self.skip_connection == 'concat':
            x1 = torch.cat([x, x1], axis=1)

        return x1

def instantiate_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_()

class SLAD(BaseDeepAD):
    def __init__(self, 
                #  seed = 0, model_name = "slad", 
                 epochs=100, batch_size=128, lr=1e-3,
                 hidden_dims=100, act='LeakyReLU',
                 distribution_size=10, # the member size in a group, c in the paper
                 n_ensemble=20,
                 subspace_pool_size=50,
                 magnify_factor=200,
                 n_unified_features=128, # dimensionality after transformation, h in the paper
                 epoch_steps=-1, prt_steps=10, 
                 device=torch.device("cpu"),
                #  device='cuda:0',
                 verbose=2, random_state=42):
        super(SLAD, self).__init__(
            model_name='SLAD', epochs=epochs, batch_size=batch_size, lr=lr,
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.act = act

        self.distribution_size = distribution_size
        self.n_ensemble = n_ensemble

        self.max_subspace_len = None # maximum length of subspace
        self.sampling_size = None # number of objects per ensemble member
        self.subspace_pool_size = subspace_pool_size

        self.n_unified_features = n_unified_features
        self.magnify_factor = magnify_factor

        self.len_pool = None
        self.affine_network_lst = {}
        self.subspace_indices_lst = []
        self.f_weight = None

        self.net = None

        return

    def training_prepare(self, X, y=None):
        self.adaptively_setting()
        if self.verbose >= 1:
            print(f'unified size: {self.n_unified_features}, '
                  f'subspace pool size: {self.subspace_pool_size}, '
                  f'ensemble size: {self.n_ensemble}')

        # randomly determines the pool of subspace sizes
        self.len_pool = np.sort(np.random.choice(np.arange(1, self.max_subspace_len+1),
                                                 self.subspace_pool_size,
                                                 replace=False))
        if self.verbose >=1:
            print('len pool:', self.len_pool)

        for s in self.len_pool:
            self.affine_network_lst[s] = LinearBlock(
                in_channels=s, out_channels=self.n_unified_features,
                bias=False, activation=None
            )

        # calculate feature weight by Pearson correlation coefficient matrix
        self.f_weight = self._cal_f_weight(X)

        # get newly generated data with supervision signals
        x_new, y_new = self._transform_data_ensemble(X)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(x_new), torch.from_numpy(y_new)),
                                  batch_size=self.batch_size, shuffle=True)

        net = MLPnet(
            n_features=self.n_unified_features,
            n_hidden=self.hidden_dims,
            n_output=1,
            activation=self.act
        ).to(self.device)

        criterion = SLADLoss()

        return train_loader, net, criterion

    def predict_score(self, X):
        criterion = SLADLoss(reduction='none')
        all_score_lst = []
        for i in range(self.n_ensemble):
            subspace_indices = self.subspace_indices_lst[i]
            x_new, y_new = self._transform_data(X,
                                               subspace_indices=subspace_indices,
                                               subset_idx=np.arange(len(X)))

            test_loader = DataLoader(TensorDataset(torch.from_numpy(x_new), torch.from_numpy(y_new)),
                                     batch_size=self.batch_size, shuffle=False, drop_last=False)

            self.net.eval()
            with torch.no_grad():
                score_lst = []
                for batch_x in test_loader:
                    s = self.inference_forward(batch_x, self.net, criterion)
                    score_lst.append(s)

            scores = torch.cat(score_lst).data.cpu().numpy()
            all_score_lst.append(scores)

        final_s = np.average(np.array(all_score_lst), axis=0)
        return final_s

    def training_forward(self, batch_x, net, criterion):
        batch_x, batch_y = batch_x
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        predict_y = net(batch_x)
        predict_y = predict_y.squeeze(dim=2)

        loss = criterion(predict_y, batch_y)

        return loss

    def inference_forward(self, batch_x, net, criterion):
        return self.training_forward(batch_x, net, criterion)

    def inference_prepare(self, X):
        pass

    def _cal_f_weight(self, X):
        if self.n_features < 50:
            pccs = np.abs(np.corrcoef(X.T))
            pccs[np.isnan(pccs)] = 0
            f_weight = np.average(pccs, axis=1)
            f_weight = (f_weight - np.min(f_weight)) / (np.max(f_weight) - np.min(f_weight))
        else:
            f_weight = np.ones(self.n_features)
        return f_weight

    def _transform_data_ensemble(self, X):
        # get newly generated data with supervisory signals
        x_new_lst = []
        y_new_lst = []
        rng = np.random.RandomState(seed=self.random_state)
        for i in range(self.n_ensemble):
            replace = True if self.n_features <= 10 else False
            subspace_indices = [
                rng.choice(np.arange(self.n_features), rng.choice(self.len_pool, 1), replace=replace)
                for _ in range(self.distribution_size)
            ]
            self.subspace_indices_lst.append(subspace_indices)

            subset_idx = rng.choice(np.arange(X.shape[0]), self.sampling_size, replace=True)

            # the size of newly generated data: [sampling_size, distribution_size(c), n_unified_features(h)]
            x_new, y_new = self._transform_data(X, subspace_indices, subset_idx)
            x_new_lst.append(x_new)
            y_new_lst.append(y_new)

        x_new = np.vstack(x_new_lst)
        y_new = np.vstack(y_new_lst)
        return x_new, y_new

    def _transform_data(self, X, subspace_indices, subset_idx):
        """generate new data sets with supervision according to subspace indices"""

        x_new = np.zeros([len(subset_idx), self.distribution_size, self.n_unified_features])
        y_new = np.zeros([len(subset_idx), self.distribution_size])

        for ii, subspace_idx in enumerate(subspace_indices):
            n_f = len(subspace_idx)

            # # get the subspace
            # # use two steps here, otherwise the output shape is 1-dim when subspace is with 1 feature
            x_sub = X[subset_idx, :]
            x_sub = x_sub[:, subspace_idx]

            # # transform function: get transformed vectors
            x_sub_projected = self._transformation_function(x_sub, n_f)

            # # use feature weight / number of features to set supervision signals
            target = (np.sum(self.f_weight[subspace_idx]) / self.n_unified_features) * self.magnify_factor
            y_true = np.array([target] * x_sub.shape[0])

            x_new[:, ii, :], y_new[:, ii] = x_sub_projected, y_true

        return x_new, y_new

    def _transformation_function(self, x, n_f):
        """transform phase to obtain a unified dimensionality"""

        # # affine transform
        transform_net = self.affine_network_lst[n_f]
        transform_net.eval()
        with torch.no_grad():
            x_projected = transform_net(torch.from_numpy(x).float()).data.cpu().numpy()
        return x_projected

    def adaptively_setting(self):
        n_samples = self.n_samples
        n_features = self.n_features

        self.max_subspace_len = n_features

        if self.subspace_pool_size is None:
            self.subspace_pool_size = min(self.max_subspace_len, 256)
        else:
            self.subspace_pool_size = min(self.subspace_pool_size, self.max_subspace_len)

        if n_samples < 500:
            factor = 50
        elif n_samples < 1000:
            factor = 20
        elif 1000 <= n_samples < 5000:
            factor = 5
        elif n_samples > 100000:
            factor = 1
        else:
            factor = 2
        self.sampling_size = max(int(n_samples / self.n_ensemble) * factor, 10)

        return


class SLADLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(SLADLoss, self).__init__()
        assert reduction in ['mean', 'none', 'sum'], 'unsupported reduction operation'

        self.reduction = reduction
        self.kl = torch.nn.KLDivLoss(reduction='none')

        return

    def forward(self, y_pred, y_true):
        """
        forward function

        Parameters:
        y_pred: torch.Tensor, shape = [batch_size, distribution_size]
            output of the network

        y_true: torch.Tensor, shape = [batch_size, distribution_size]
            ground truth labels

        return_raw_results: bool, default=False
            return the raw results with shape [batch_size, distribution_size]
            accompanied by reduced loss value

        Return:

        loss value: torch.Tensor
            reduced loss value
        """

        reduction = self.reduction

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        total_m = 0.5 * (preds_smax + true_smax)

        js1 = F.kl_div(F.log_softmax(preds_smax, dim=1), total_m, reduction='none')
        js2 = F.kl_div(F.log_softmax(true_smax, dim=1), total_m, reduction='none')
        js = torch.sum(js1 + js2, dim=1)

        if reduction == 'mean':
            loss = torch.mean(js)
        elif reduction == 'sum':
            loss = torch.sum(js)
        elif reduction == 'none':
            loss = js
        else:
            return

        return loss