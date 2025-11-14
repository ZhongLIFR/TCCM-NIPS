"""
MCM: Adapted from its official implementation (https://openreview.net/forum?id=lNZJyEDxy4)

"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
# import copy

def get_device(gpu_specific=False):
    if gpu_specific:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f'number of gpu: {n_gpu}')
            print(f'cuda name: {torch.cuda.get_device_name(0)}')
            print('GPU is on')
        else:
            print('GPU is off')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


class Generator(nn.Module):
    def __init__(self, model, config):
        super(Generator, self).__init__()
        self.masks = model._make_nets(config['data_dim'], config['mask_nlayers'], config['mask_num'])
        self.mask_num = config['mask_num']
        self.device = config['device']

    def forward(self, x):
        x = x.type(torch.FloatTensor).to(self.device)
        x_T = torch.empty(x.shape[0], self.mask_num, x.shape[-1]).to(x)
        masks = []
        for i in range(self.mask_num):
            mask = self.masks[i](x)
            masks.append(mask.unsqueeze(1))
            mask = torch.sigmoid(mask)
            x_T[:, i] = mask * x
        masks = torch.cat(masks, axis=1)
        return x_T, masks


class SingleNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super(SingleNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim,h_dim,bias=False))
            net.append(nn.ReLU())
            input_dim= h_dim
        net.append(nn.Linear(input_dim,x_dim,bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class MultiNets():
    def _make_nets(self, x_dim, mask_nlayers, mask_num):
        multinets = nn.ModuleList(
            [SingleNet(x_dim, x_dim, mask_nlayers) for _ in range(mask_num)])
        return multinets


class LossFunction(nn.Module):
    def __init__(self, model_config):
        super(LossFunction, self).__init__()
        self.mask_num = model_config['mask_num']
        self.divloss = DiversityMask()
        self.lamb = model_config['lambda']

    def forward(self, x_input, x_pred, masks):
        x_input = x_input.unsqueeze(1).repeat(1, self.mask_num, 1)
        sub_result = x_pred - x_input
        mse = torch.norm(sub_result, p=2, dim=2)
        mse_score = torch.mean(mse, dim=1, keepdim=True)
        e = torch.mean(mse_score)
        divloss = self.divloss(masks)
        loss = torch.mean(e) + self.lamb*torch.mean(divloss)
        return loss, torch.mean(e), torch.mean(divloss)


class DiversityMask(nn.Module):
    def __init__(self,temperature=0.1):
        super(DiversityMask, self).__init__()
        self.temp = temperature

    def forward(self,z,eval=False):
        z = F.normalize(z, p=2, dim=-1)
        batch_size, mask_num, z_dim = z.shape
        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(mask_num).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, mask_num, -1)
        trans_matrix = sim_matrix.sum(-1)
        if mask_num > 2:
            K = mask_num - 1
        else:
            K = mask_num
        scale = 1 / np.abs(K*np.log(1.0 / K))
        loss_tensor = torch.log(trans_matrix) * scale
        if eval:
            score = loss_tensor.sum(1)
            return score
        else:
            loss = loss_tensor.sum(1)
            return loss


class ScoreFunction(nn.Module):
    def __init__(self, model_config):
        super(ScoreFunction, self).__init__()
        self.mask_num = model_config['mask_num']

    def forward(self, x_input, x_pred):
        x_input = x_input.unsqueeze(1).repeat(1, self.mask_num, 1)
        sub_result = x_pred - x_input
        mse = torch.norm(sub_result, p=2, dim=2)
        mse_score = torch.mean(mse, dim=1,keepdim=True)
        return mse_score
    

class MCMModel(nn.Module):
    def __init__(self, model_config):
        super(MCMModel, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']
        self.z_dim = model_config['z_dim']
        self.mask_num = model_config['mask_num']
        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.maskmodel = Generator(MultiNets(), model_config)

        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim

        encoder.append(nn.Linear(encoder_dim,self.z_dim,bias=False))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder_dim = self.z_dim
        for _ in range(self.de_nlayers-1):
            decoder.append(nn.Linear(decoder_dim,self.hidden_dim,bias=False))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = self.hidden_dim

        decoder.append(nn.Linear(decoder_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        x_mask, masks = self.maskmodel(x_input)
        B, T, D = x_mask.shape
        x_mask = x_mask.reshape(B*T, D)
        z = self.encoder(x_mask)
        x_pred = self.decoder(z)
        z = z.reshape(x_input.shape[0], self.mask_num, z.shape[-1])
        x_pred = x_pred.reshape(x_input.shape[0], self.mask_num, x_input.shape[-1])
        return x_pred, z, masks

    def print_weight(self, x_input):
        x_input = Variable(x_input, requires_grad=False)
        z = self.encoder(x_input)
        fea_mem = self.fea_mem(z)
        fea_att_w = fea_mem['att']
        out = torch.max(fea_att_w, dim=0).view(8, 8).detach().cpu().numpy()
        return out
    
class MCM():
    def __init__(self, n_features,
                 batch_size = 256, # Batch size of training
                 epochs = 200, # Epoch size of training,
                 learning_rate = 0.05, # Learning rate of training,
                 test_batch_size = 32,  # As defined in the original implementation
                 device = get_device(False)
                 ):
        # self.masks = model._make_nets(config['data_dim'], config['mask_nlayers'], config['mask_num'])
        # self.mask_num = config['mask_num']
        # self.device = config['device']
        self.device = device
        self.data_dim = n_features
        self.model_config = {
            "data_dim": n_features,
            "hidden_dim": 256,
            "z_dim": 128,
            "mask_num": 50,
            "mask_nlayers": 3,
            "en_nlayers": 3,
            "de_nlayers": 3,
            "lambda": 5,
            "device": device,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "sche_gamma": 0.98,
            }
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs

        # self.train_loader, self.test_loader, self.data_shape = get_dataloader(model_config, path)
        if n_features > 1000:
            self.model_config['mask_num'] = n_features//6
        else:
            self.model_config['mask_num'] = n_features//2
        # self.model = MCMModel(model_config).to(self.device)
        self.model = None
        self.loss_fuc = LossFunction(self.model_config).to(self.device)
        self.score_func = ScoreFunction(self.model_config).to(self.device)

    def fit(self, X_train, y_train=None):
        X = torch.tensor(X_train, dtype=torch.float32)
        if y_train is None:
            y_train = torch.zeros(X.shape[0], dtype=torch.long).squeeze() # We assume semi-supervised manner
        train_loader = DataLoader(TensorDataset(X, y_train), batch_size=self.batch_size, shuffle=True)

        self.model = MCMModel(self.model_config).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.model_config['sche_gamma'])

        self.model.train()
        min_loss = 100
        self.good_model = None
        for epoch in range(self.epochs):
            for step, (x_input, y_label) in enumerate(train_loader):
                # print(x_input.shape)
                # print(y_label.shape)
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            # info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
            # train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
            if loss < min_loss:
            #     torch.save(self.model, 'model.pth') # This creates so many overheads...
                self.model.eval()
                # self.good_model = copy.deepcopy(self.model) # Always save the newest checkpoint
                with torch.no_grad(): # Always save the newest checkpoint
                    good_model = MCMModel(self.model_config).to(self.device)
                    good_model.load_state_dict(self.model.state_dict())
                self.model.train()
                min_loss = loss
        # if self.good_model is None:
        #     self.good_model = self.model
        # self.model = None
        assert good_model is not None
        self.model = good_model
        # print("Training complete.")

    # @torch.no_grad
    def decision_function(self, X_test):
        self.model.eval()
        mse_score, test_label = [], []

        X = torch.tensor(X_test, dtype=torch.float32)
        X = X.to(next(self.model.parameters()).device)
        y_test_dummy = torch.zeros(X.shape[0], dtype=torch.long).squeeze() # We assume semi-supervised manner
        test_loader = DataLoader(TensorDataset(X, y_test_dummy), batch_size=self.test_batch_size, shuffle=False)

        with torch.no_grad():
            for step, (x_input, y_label) in enumerate(test_loader):
                x_input = x_input.to(self.device)
                x_pred, z, masks = self.model(x_input)
                mse_batch = self.score_func(x_input, x_pred)
                mse_batch = mse_batch.data.cpu()
                mse_score.append(mse_batch)
                # test_label.append(y_label)
        mse_score = torch.cat(mse_score, axis=0).numpy()
        # test_label = torch.cat(test_label, axis=0).numpy()
        # mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
        # mse_f1[self.run] = F1Performance(mse_score, test_label)
        # print(self.train_loader.shape)
        return mse_score


# class Trainer(object):
#     def __init__(self, run: int, model_config: dict, path: str):
#         self.run = run
#         # self.sche_gamma = model_config['sche_gamma']
#         # self.device = model_config['device']
#         # self.learning_rate = model_config['learning_rate']
#         self.train_loader, self.test_loader, self.data_shape = get_dataloader(model_config, path)
#         model_config['data_dim'] = self.data_shape[1]
#         if self.data_shape[1] > 1000:
#             model_config['mask_num'] = self.data_shape[1]//6
#         else:
#             model_config['mask_num'] = self.data_shape[1]//2
#         self.model = MCMModel(model_config).to(self.device)
#         self.loss_fuc = LossFunction(model_config).to(self.device)
#         self.score_func = ScoreFunction(model_config).to(self.device)
#         # self.train_loader, self.test_loader = get_dataloader(model_config)

#     def training(self, epochs):
#         # train_logger = get_logger('train_log.log')
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
#         scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
#         self.model.train()
#         # print(self.model)
#         print("Training Start.")
#         min_loss = 100
#         for epoch in range(epochs):
#             for step, (x_input, y_label) in enumerate(self.train_loader):
#                 # print(x_input.shape)
#                 # print(y_label.shape)
#                 x_input = x_input.to(self.device)
#                 x_pred, z, masks = self.model(x_input)
#                 loss, mse, divloss = self.loss_fuc(x_input, x_pred, masks)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             scheduler.step()
#             # info = 'Epoch:[{}]\t loss={:.4f}\t mse={:.4f}\t divloss={:.4f}\t'
#             # train_logger.info(info.format(epoch,loss.cpu(),mse.cpu(),divloss.cpu()))
#             # if loss < min_loss:
#             #     torch.save(self.model, 'model.pth')
#             #     min_loss = loss
#         # print("Training complete.")
#         # train_logger.handlers.clear()

#     def evaluate(self, mse_rauc, mse_ap, mse_f1):
#         model = torch.load('model.pth')
#         model.eval()
#         mse_score, test_label = [], []
#         for step, (x_input, y_label) in enumerate(self.test_loader):
#             x_input = x_input.to(self.device)
#             x_pred, z, masks = self.model(x_input)
#             mse_batch = self.score_func(x_input, x_pred)
#             mse_batch = mse_batch.data.cpu()
#             mse_score.append(mse_batch)
#             test_label.append(y_label)
#         mse_score = torch.cat(mse_score, axis=0).numpy()
#         test_label = torch.cat(test_label, axis=0).numpy()
#         # mse_rauc[self.run], mse_ap[self.run] = aucPerformance(mse_score, test_label)
#         # mse_f1[self.run] = F1Performance(mse_score, test_label)
#         # print(self.train_loader.shape)
#         return mse_score