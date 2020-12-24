# The code is based on original repository https://github.com/OctoberChang/klcpd_code
#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import cPickle as pickle
import math
import numpy as np
import os
import random
import sklearn.metrics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from sklearn.metrics.pairwise import euclidean_distances

from data_loader import DataLoader
from optim import Optim


def median_heuristic(X, beta=0.5):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    beta_list = [beta**2, beta**1, 1, (1.0/beta)**1, (1.0/beta)**2]
    return [med_sqdist * b for b in beta_list]


# X_p_enc: batch_size x seq_len x hid_dim
# X_f_enc: batch_size x seq_len x hid_dim
# hid_dim could be either dataspace_dim or codespace_dim
# return: MMD2(X_p_enc[i,:,:], X_f_enc[i,:,:]) for i = 1:batch_size
def batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var):
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
        U = torch.cuda.FloatTensor(batch_size*n_samples, n_mixtures).uniform_()
        sigma_samples = F.softmax(U * gumbel_lmd).matmul(sigma_var)
        W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
        W_gmm = W_gmm.view(batch_size, n_samples, nz)
        return W_gmm

    W = Variable(torch.cuda.FloatTensor(batch_size*n_samples, nz).normal_(0, 1))
    W_gmm = sample_gmm(W, batch_size)                                   # batch_size x n_samples x nz
    W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()                   # batch_size x nz x n_samples
    XW_p = torch.bmm(X_p_enc, W_gmm)                                    # batch_size x seq_len x n_samples
    XW_f = torch.bmm(X_f_enc, W_gmm)                                    # batch_size x seq_len x n_samples
    z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
    z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
    batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1))**2, 1)
    return batch_mmd2_rff




class NetG(nn.Module):
    def __init__(self, args, data):
        super(NetG, self).__init__()
        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, num_layers=1, batch_first=True)
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
    def __init__(self, args, data):
        super(NetD, self).__init__()

        self.wnd_dim = args.wnd_dim
        self.var_dim = data.var_dim
        self.D = data.D
        self.RNN_hid_dim = args.RNN_hid_dim

        self.rnn_enc_layer = nn.GRU(self.var_dim, self.RNN_hid_dim, batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.var_dim, batch_first=True)

    def forward(self, X):
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        return X_enc, X_dec


# Y, L should be numpy array
def valid_epoch(loader, data, netD, batch_size, Y_true, L_true):
    netD.eval()
    Y_pred = []
    for inputs in loader.get_batches(data, batch_size, shuffle=False):
        X_p, X_f = inputs[0], inputs[1]
        batch_size = X_p.size(0)

        X_p_enc, _ = netD(X_p)
        X_f_enc, _ = netD(X_f)
        Y_pred_batch = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)
        Y_pred.append(Y_pred_batch.data.cpu().numpy())
    Y_pred = np.concatenate(Y_pred, axis=0)

    L_pred = Y_pred
    fp_list, tp_list, thresholds = sklearn.metrics.roc_curve(L_true, L_pred)
    auc = sklearn.metrics.auc(fp_list, tp_list)
    eval_dict = {'Y_pred': Y_pred,
                 'L_pred': L_pred,
                 'Y_true': Y_true,
                 'L_true': L_true,
                 'mse': -1, 'mae': -1, 'auc': auc}
    return eval_dict

default_args = {
    'window_size': 10,
    'sub_dim': 1,
    'RNN_hid_dim': 10,
    'batch_size': 128,
    'max_iter': 100,
    'lr': 3e-4,
    'weight_decay': 0.,
    'momentum': 0.,
    'grad_clip': 10.,
    'CRITIC_ITERS': 5,
    'weight_clip': .1,
    'lambda_ae': 0.001,
    'lambda_real': 0.1
}




# ========= Load Dataset and initialize model=========#
Data = DataLoader(args, trn_ratio=args.trn_ratio, val_ratio=args.val_ratio)
netG = NetG(args, Data)
netD = NetD(args, Data)

netG_params_count = sum([p.nelement() for p in netG.parameters()])
netD_params_count = sum([p.nelement() for p in netD.parameters()])

one = torch.cuda.FloatTensor([1])
mone = one * -1


# ========= Setup loss function and optimizer  =========#
optimizerG = Optim(netG.parameters(),
                   args.optim,
                   lr=args.lr,
                   grad_clip=args.grad_clip,
                   weight_decay=args.weight_decay,
                   momentum=args.momentum)

optimizerD = Optim(netD.parameters(),
                   args.optim,
                   lr=args.lr,
                   grad_clip=args.grad_clip,
                   weight_decay=args.weight_decay,
                   momentum=args.momentum)


# sigma for mixture of RBF kernel in MMD
#sigma_list = [1.0]
#sigma_list = median_heuristic(Data.Y_subspace, beta=1.)
sigma_list = median_heuristic(Data.Y_subspace, beta=.5)
sigma_var = torch.FloatTensor(sigma_list).cuda()
print('sigma_list:', sigma_var)


# ========= Main loop for adversarial training kernel with negative samples X_f + noise =========#
Y_val = Data.val_set['Y'].numpy()
L_val = Data.val_set['L'].numpy()
Y_tst = Data.tst_set['Y'].numpy()
L_tst = Data.tst_set['L'].numpy()

n_batchs = int(math.ceil(len(Data.trn_set['Y']) / float(args.batch_size)))
print('n_batchs', n_batchs, 'batch_size', args.batch_size)

lambda_ae = args.lambda_ae
lambda_real = args.lambda_real
gen_iterations = 0
total_time = 0.
best_epoch = -1
best_val_mae = 1e+6
best_val_auc = -1
best_tst_auc = -1
best_mmd_real = 1e+6
start_time = time.time()
print('start training: lambda_ae', lambda_ae, 'lambda_real', lambda_real, 'weight_clip', args.weight_clip)
for epoch in range(1, args.max_iter + 1):
    trn_loader = Data.get_batches(Data.trn_set, batch_size=args.batch_size, shuffle=True)
    bidx = 0
    while bidx < n_batchs:
        ############################
        # (1) Update D network
        ############################
        for p in netD.parameters():
            p.requires_grad = True

        for diters in range(args.CRITIC_ITERS):
            # clamp parameters of NetD encoder to a cube
            for p in netD.rnn_enc_layer.parameters():
                p.data.clamp_(-args.weight_clip, args.weight_clip)
            if bidx == n_batchs:
                break

            inputs = next(trn_loader)
            X_p, X_f, Y_true = inputs[0], inputs[1], inputs[2]
            batch_size = X_p.size(0)
            bidx += 1

            # real data
            X_p_enc, X_p_dec = netD(X_p)
            X_f_enc, X_f_dec = netD(X_f)

            # fake data
            noise = torch.cuda.FloatTensor(1, batch_size, args.RNN_hid_dim).normal_(0, 1)
            noise = Variable(noise, volatile=True) # total freeze netG
            Y_f = Variable(netG(X_p, X_f, noise).data)
            Y_f_enc, Y_f_dec = netD(Y_f)

            # batchwise MMD2 loss between X_f and Y_f
            D_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)

            # batchwise MMD loss between X_p and X_f
            mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)

            # reconstruction loss
            real_L2_loss = torch.mean((X_f - X_f_dec)**2)
            #real_L2_loss = torch.mean((X_p - X_p_dec)**2)
            fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2)
            #fake_L2_loss = torch.mean((Y_f - Y_f_dec)**2) * 0.0

            # update netD
            netD.zero_grad()
            lossD = D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()
            #lossD = 0.0 * D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()
            #lossD = -real_L2_loss
            lossD.backward(mone)
            optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        if bidx == n_batchs:
            break

        inputs = next(trn_loader)
        X_p, X_f = inputs[0], inputs[1]
        batch_size = X_p.size(0)
        bidx += 1

        # real data
        X_f_enc, X_f_dec = netD(X_f)

        # fake data
        noise = torch.cuda.FloatTensor(1, batch_size, args.RNN_hid_dim).normal_(0, 1)
        noise = Variable(noise)
        Y_f = netG(X_p, X_f, noise)
        Y_f_enc, Y_f_dec = netD(Y_f)

        # batchwise MMD2 loss between X_f and Y_f
        G_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)

        # update netG
        netG.zero_grad()
        lossG = G_mmd2.mean()
        #lossG = 0.0 * G_mmd2.mean()
        lossG.backward(one)
        optimizerG.step()

        #G_mmd2 = Variable(torch.FloatTensor(batch_size).zero_())
        gen_iterations += 1

        print('[%5d/%5d] [%5d/%5d] [%6d] D_mmd2 %.4e G_mmd2 %.4e mmd2_real %.4e real_L2 %.6f fake_L2 %.6f'
              % (epoch, args.max_iter, bidx, n_batchs, gen_iterations,
                 D_mmd2.mean().data[0], G_mmd2.mean().data[0], mmd2_real.mean().data[0],
                 real_L2_loss.data[0], fake_L2_loss.data[0]))

        if gen_iterations % args.eval_freq == 0:
            # ========= Main block for evaluate MMD(X_p_enc, X_f_enc) on RNN codespace  =========#
            val_dict = valid_epoch(Data, Data.val_set, netD, args.batch_size, Y_val, L_val)
            tst_dict = valid_epoch(Data, Data.tst_set, netD, args.batch_size, Y_tst, L_tst)
            total_time = time.time() - start_time
            print('iter %4d tm %4.2fm val_mse %.1f val_mae %.1f val_auc %.6f'
                    % (epoch, total_time / 60.0, val_dict['mse'], val_dict['mae'], val_dict['auc']), end='')

            print (" tst_mse %.1f tst_mae %.1f tst_auc %.6f" % (tst_dict['mse'], tst_dict['mae'], tst_dict['auc']), end='')

            assert(np.isnan(val_dict['auc']) != True)
            #if val_dict['auc'] > best_val_auc:
            #if val_dict['auc'] > best_val_auc and mmd2_real.mean().data[0] < best_mmd_real:
            if mmd2_real.mean().data[0] < best_mmd_real:
                best_mmd_real = mmd2_real.mean().data[0]
                best_val_mae = val_dict['mae']
                best_val_auc = val_dict['auc']
                best_tst_auc = tst_dict['auc']
                best_epoch = epoch
                save_pred_name = '%s/pred.pkl' % (args.save_path)
                with open(save_pred_name, 'wb') as f:
                    pickle.dump(tst_dict, f)
                torch.save(netG.state_dict(), '%s/netG.pkl' % (args.save_path))
                torch.save(netD.state_dict(), '%s/netD.pkl' % (args.save_path))
            print(" [best_val_auc %.6f best_tst_auc %.6f best_epoch %3d]" % (best_val_auc, best_tst_auc, best_epoch))

        # stopping condition
        #if best_mmd_real < 1e-4:
        if mmd2_real.mean().data[0] < 1e-5:
            exit(0)
