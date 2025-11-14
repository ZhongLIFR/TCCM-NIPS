"""
Adapted from the On Diffusion Modeling for Anomaly Detection (https://openreview.net/forum?id=lR3rk7ysXz&noteId=lR3rk7ysXz)

"""
import numpy as np
import torch
import torch.distributions as dist

from sklearn.neighbors import NearestNeighbors, BallTree

from scipy.stats import invgamma


# semi supervised train test split
def binning(t, T,  num_bins=30):
    return torch.maximum(torch.minimum(torch.floor(t*num_bins/T), torch.tensor(num_bins-1)), torch.tensor(0)).long()

def create_noisy_data(X, noise_std):
    noise = torch.randn_like(X) * noise_std
    return X + noise

def compute_pairwise_diff(X1, X2):
    #return torch.sqrt(torch.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    return X1[:, None, :] - X2[None, :, :]

def train_test_split_anomaly(X, y, train_split=0.5):
    indices = np.arange(len(X))
    normal_indices = indices[y == 0]
    anomaly_indices = indices[y == 1]

    train_size = round(train_split * normal_indices.size)
    train_indices, test_indices = normal_indices[:train_size], normal_indices[train_size:]
    test_indices = np.append(test_indices, anomaly_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

class DTENonParametric(object):
    def __init__(self, 
                #  seed = 0, model_name = "DTE-NP", 
                 K=5, T=1000):
        beta_0 = 0.0001
        beta_T = 0.01
        self.T = T
        self.K = K
        # self.seed = seed
        self.T_range = np.arange(0, self.T)
        betas = torch.linspace(beta_0, beta_T, self.T)
        
        self.neigh = NearestNeighbors(n_neighbors=K,
                                       radius=1.0,
                                       algorithm='auto',
                                       leaf_size=30,
                                       metric='minkowski',
                                       p=2,
                                       metric_params=None,
                                       n_jobs=1)
        

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) #std deviations
        # self.model_name = model_name

    def compute_log_likelihood(self, X):
        N1, N2, dim = X.shape
        log_likelihood = torch.zeros((self.T, N1, N2))
        # loop because one shotting causes memory issues
        for t in range(self.T):
            loc = torch.zeros((dim))
            scale = torch.ones((dim)) * self.sqrt_one_minus_alphas_cumprod[t]
            dist_t = dist.Independent(dist.Normal(loc=loc, scale=scale), 1)
            #dist_t = dist.Normal(loc=0., scale=sqrt_one_minus_alphas_cumprod[t])
            log_likelihood[t, ...] = dist_t.log_prob(X)
        return log_likelihood

    def kernel_estimator(self, X_test, timestep=0, eval=False):
        _, dim = X_test.shape
        X_test = torch.from_numpy(X_test).float()
        if eval:
            X_noisy = X_test.clone()
        else:
            X_noisy = create_noisy_data(X_test, self.sqrt_one_minus_alphas_cumprod[timestep])

        log_p_t_given_y = torch.zeros((self.T, X_test.shape[0]))
    
        # non-parametric solution    
        min_norm_2 = np.zeros([X_test.shape[0], 1])

        for i in range(X_test.shape[0]):
            x_i = X_test[i, :]
            x_i = np.asarray(x_i).reshape(1, x_i.shape[0])

            # get the distance of the current point
            dist_arr, _ = self.tree.query(x_i, k=self.K)
            dist = np.mean(dist_arr, -1)
            min_norm_2[i, :] = dist[-1]

        density = torch.zeros((self.T, X_test.shape[0]))
        for i in range(min_norm_2.shape[0]):
            density[:,i] = torch.from_numpy(invgamma.logpdf((1. - self.alphas_cumprod), a=0.5*dim-1, \
                                loc=0, scale=(min_norm_2[i]/2))).float()
        
        density = density - density.logsumexp(0, keepdim=True)

        return log_p_t_given_y.exp().t(), density.exp().t()
    

    def nonparametric(self, X_test, timestep=0, eval=False):

        p_t_given_y, density = self.kernel_estimator(X_test, timestep=timestep, eval=eval)
        
        return p_t_given_y, density

    def compute_timestep_prediction(self, X_test, X_train):
        p_t = torch.zeros((self.T, X_test.shape[0], self.T))
        invgamma_p_t = torch.zeros((self.T, X_test.shape[0], self.T))
        for t in range(self.T):
            p_t[t, ...], invgamma_p_t[t, ...] = self.kernel_estimator(X_test, X_train, timestep=t)
            print('Completed t = {}/{}'.format(t, self.T), end='\r')
        print('\n')

        np.save('./{}_p_t.npy'.format(self.dataset_name), p_t.numpy())
        np.save('./{}_invgamma_p_t.npy'.format(self.dataset_name), invgamma_p_t.numpy())
        return

    def fit(self, X_train, y_train=None):
        self.neigh.fit(X_train)
        
        if self.neigh._tree is not None:
            self.tree = self.neigh._tree
        else:
            self.tree = BallTree(X_train, leaf_size=30, metric='minkowski')
        
        return self

    def predict_score(self, X_test):
        p_t, invgamma_p_t = self.nonparametric(X_test, timestep=0, eval=True)
        
        preds = torch.argmax(invgamma_p_t,axis=-1).float().numpy()

        return preds
