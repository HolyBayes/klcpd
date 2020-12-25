# The code is based on original repository https://github.com/OctoberChang/klcpd_code
#!/usr/bin/env python
# encoding: utf-8

import math
import numpy as np
import random
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.metrics.pairwise import euclidean_distances

from .data import HankelDataset
from types import SimpleNamespace
from tqdm import trange
from torch.utils.data import DataLoader


def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]

class NetG(nn.Module):
    def __init__(self, var_dim, RNN_hid_dim, num_layers:int=1):
        super().__init__()
        self.var_dim = var_dim
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, self.var_dim)

    # X_p:   batch_size x wnd_dim x var_dim (Encoder input)
    # X_f:   batch_size x wnd_dim x var_dim (Decoder input)
    # h_t:   1 x batch_size x RNN_hid_dim
    # noise: 1 x batch_size x RNN_hid_dim
    def forward(self, X_p, X_f, noise):
        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X):
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft


class NetD(nn.Module):
    def __init__(self, var_dim, RNN_hid_dim, num_layers:int=1):
        super(NetD, self).__init__()

        self.var_dim = var_dim
        self.RNN_hid_dim = RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=num_layers, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, num_layers=num_layers, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


class KL_CPD(nn.Module):
    def __init__(self, D:int, critic_iters:int=5,
            lambda_ae:float=0.001, lambda_real:float=0.1,
            p_wnd_dim:int=25, f_wnd_dim:int=10, sub_dim:int=1, RNN_hid_dim:int=10):
        super().__init__()
        self.p_wnd_dim = p_wnd_dim
        self.f_wnd_dim = f_wnd_dim
        self.sub_dim = sub_dim
        self.D = D
        self.var_dim = D * sub_dim
        self.critic_iters = critic_iters
        self.lambda_ae, self.lambda_real = lambda_ae, lambda_real
        self.RNN_hid_dim = RNN_hid_dim
        self.netD = NetD(self.var_dim, RNN_hid_dim)
        self.netG = NetG(self.var_dim, RNN_hid_dim)


    @property
    def device(self):
        return next(self.parameters()).device

    def __mmd2_loss(self, X_p_enc, X_f_enc):
        sigma_var = self.sigma_var

        # some constants
        n_basis = 1024
        gumbel_lmd = 1e+6
        cnst = math.sqrt(1. / n_basis)
        n_mixtures = sigma_var.size(0)
        n_samples = n_basis * n_mixtures
        batch_size, seq_len, nz = X_p_enc.size()

        # gumbel trick to get masking matrix to uniformly sample sigma
        # input: (batch_size*n_samples, nz)
        # output: (batch_size, n_samples, nz)
        def sample_gmm(W, batch_size):
            U = torch.FloatTensor(batch_size*n_samples, n_mixtures).uniform_().to(self.device)
            sigma_samples = F.softmax(U * gumbel_lmd).matmul(sigma_var)
            W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
            W_gmm = W_gmm.view(batch_size, n_samples, nz)
            return W_gmm

        W = Variable(torch.FloatTensor(batch_size*n_samples, nz).normal_(0, 1).to(self.device))
        W_gmm = sample_gmm(W, batch_size)                                   # batch_size x n_samples x nz
        W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()                   # batch_size x nz x n_samples
        XW_p = torch.bmm(X_p_enc, W_gmm)                                    # batch_size x seq_len x n_samples
        XW_f = torch.bmm(X_f_enc, W_gmm)                                    # batch_size x seq_len x n_samples
        z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
        z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
        batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1))**2, 1)
        return batch_mmd2_rff

    def forward(self, X_p:torch.Tensor, X_f:torch.Tensor):
        batch_size = X_p.size(0)

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        Y_pred_batch = self.__mmd2_loss(X_p_enc, X_f_enc)

        return Y_pred_batch

    def predict(self, ts):
        dataset = HankelDataset(ts, self.p_wnd_dim, self.f_wnd_dim, self.sub_dim)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in dataloader:
                X_p, X_f = [batch[key].float().to(self.device) for key in ['X_p', 'X_f']]
                preds.append(self.forward(X_p, X_f).detach().cpu().numpy())
        return np.concatenate(preds)



    def fit(self, ts, epoches:int=100,lr:float=3e-4,weight_clip:float=.1,weight_decay:float=0.,momentum:float=0.):
        # must be defined in fit() method
        optG = torch.optim.AdamW(self.netG.parameters(),lr=lr,weight_decay=weight_decay)

        optD = torch.optim.AdamW(self.netD.parameters(),lr=lr,weight_decay=weight_decay)


        dataset = HankelDataset(ts, self.p_wnd_dim, self.f_wnd_dim, self.sub_dim)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        sigma_list = median_heuristic(dataset.Y_hankel, beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list).to(self.device)


        tbar = trange(epoches)
        for epoch in tbar:
            for batch in dataloader:
                # Fit critic
                for p in self.netD.parameters():
                    p.requires_grad = True
                for p in self.netD.rnn_enc_layer.parameters():
                    p.data.clamp_(-weight_clip, weight_clip)
                self._optimizeD(batch, optD)
                if np.random.choice(np.arange(self.critic_iters)) == 0:
                    # Fit generator
                    for p in self.netD.parameters():
                        p.requires_grad = False  # to avoid computation
                    self._optimizeG(batch, optG)


    def _optimizeG(self, batch, opt, grad_clip:int=10):
        X_p, X_f = [batch[key].float().to(self.device) for key in ['X_p', 'X_f']]
        batch_size = X_p.size(0)

        # real data
        X_f_enc, X_f_dec = self.netD(X_f)

        # fake data
        noise = torch.FloatTensor(1, batch_size, self.RNN_hid_dim).normal_(0, 1).to(self.device)
        noise = Variable(noise)
        Y_f = self.netG(X_p, X_f, noise)
        Y_f_enc, Y_f_dec = self.netD(Y_f)

        # batchwise MMD2 loss between X_f and Y_f
        G_mmd2 = self.__mmd2_loss(X_f_enc, Y_f_enc)

        # update netG
        self.netG.zero_grad()
        lossG = G_mmd2.mean()
        #lossG = 0.0 * G_mmd2.mean()
        lossG.backward()

        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), grad_clip)

        opt.step()


    def _optimizeD(self, batch, opt, grad_clip:int=10):
        X_p, X_f, Y_true = [batch[key].float().to(self.device) for key in ['X_p', 'X_f', 'Y']]
        batch_size = X_p.size(0)

        # real data
        X_p_enc, X_p_dec = self.netD(X_p)
        X_f_enc, X_f_dec = self.netD(X_f)

        # fake data
        noise = torch.FloatTensor(1, batch_size, self.netG.RNN_hid_dim).normal_(0, 1).to(self.device)
        noise = Variable(noise, volatile=True) # total freeze netG
        Y_f = Variable(self.netG(X_p, X_f, noise).data)
        Y_f_enc, Y_f_dec = self.netD(Y_f)

        # batchwise MMD2 loss between X_f and Y_f
        D_mmd2 = self.__mmd2_loss(X_f_enc, Y_f_enc)

        # batchwise MMD loss between X_p and X_f
        mmd2_real = self.__mmd2_loss(X_p_enc, X_f_enc)

        # reconstruction loss
        real_L2_loss = torch.mean((X_f - X_f_dec)**2)
        fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)

        # update netD
        self.netD.zero_grad()
        lossD = D_mmd2.mean() - self.lambda_ae * (real_L2_loss + fake_L2_loss) - self.lambda_real * mmd2_real.mean()
        lossD = -lossD
        lossD.backward()

        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), grad_clip)

        opt.step()

if __name__ == '__main__':
    dim, seq_length = 1, 100
    ts = np.random.randn(seq_length,dim)
    device = torch.device('cuda')
    model = KL_CPD(dim).to(device)
    model.fit(ts)
    preds = model.predict(ts)
    print(preds)
