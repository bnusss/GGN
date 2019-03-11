#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

from utils.util import get_offdiag

use_cuda = torch.cuda.is_available()
np.random.seed(2050)
torch.manual_seed(2050)
if use_cuda:
    torch.cuda.manual_seed(2050)

# 动力学学习器dynamics_learner的一步训练
# relations的格式为num_nodes, num_nodes
# data格式为：batch_size, num_nodes, time_steps(10), dimension(4)
def train_dynamics_learner(optimizer, dynamics_learner, relations, data, sz, steps,skip_conn=False):
    # dynamics_learner.train()
    optimizer.zero_grad()
    
    adjs = relations.unsqueeze(0)
    adjs = adjs.repeat(data.size()[0],1,1)
    adjs = adjs.cuda() if use_cuda else adjs
    
    input = data[:, :, 0, :]
    target = data[:, :, 1 : steps, :]
    output = input
    
    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    # 完成steps-1步预测，output格式为：batchsize, num_nodes, time_steps, dimension
    for t in range(steps - 1):
        output = dynamics_learner(output, adjs,skip_conn)
        outputs[:,:,t,:] = output
    
    loss = torch.mean(torch.abs(outputs - target))
    loss.backward()
    optimizer.step()
    mse = F.mse_loss(outputs, target)
    if use_cuda:
        loss = loss.cpu()
        mse = mse.cpu()
    return loss, mse
# 动力学学习器dynamics_learner的一步校验

def val_dynamics_learner(dynamics_learner, relations, sz, data, steps,skip_conn=False):
    # dynamics_learner.test()

    edges = relations.float()
    adjs = relations.unsqueeze(0)
    adjs = adjs.repeat(data.size()[0],1,1)
    adjs = adjs.cuda() if use_cuda else adjs
    
    input = data[:, :, 0, :]
    target = data[:, :, 1 : steps, :]
    output = input
    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    for t in range(steps - 1):
        output = dynamics_learner(output, adjs,skip_conn)
        outputs[:,:,t,:] = output
        
    loss = torch.mean(torch.abs(outputs - target))
    mse = F.mse_loss(outputs, target)
    
    return loss, mse

# 网络生成器的一步训练
# data格式为：batch_size, num_nodes, time_steps(10), dimension(4)
# gumbel_generator为生成器，dynamics_learner为动力学预测器
def train_net_reconstructor(optimizer_network, gumbel_generator, dynamics_learner,
                            sz, data, steps,skip_conn=False):
    optimizer_network.zero_grad()

    out_matrix = gumbel_generator.sample()
    out_matrix = out_matrix.unsqueeze(0)
    out_matrix = out_matrix.repeat(data.size()[0], 1, 1)
    gumbel_generator.drop_temperature()
    losses = 0
        
    input = data[:, :, 0, :]
    target = data[:, :, 1 : steps, :]
    output = input
    outputs = torch.zeros(data.size()[0], data.size()[1], steps - 1, data.size(3))
    outputs = outputs.cuda() if use_cuda else outputs
    #做steps-1步预测
    for t in range(steps - 1):
        output = dynamics_learner(output, out_matrix,skip_conn)
        outputs[:,:,t,:] = output
        
    loss = torch.mean(torch.abs(outputs - target))
    loss.backward()
    
    optimizer_network.step()
    loss = loss.cpu() if use_cuda else loss
    return loss, out_matrix

# 让网络生成器生成tests个网络，针对每一个网络计算与目标网络obj_matrix的差异成程度
def constructor_evaluator(gumbel_generator, tests, obj_matrix, sz):
    errs= []
    tprs = []
    fprs = []

    for t in range(tests):
        # 计算net error
        out_matrix = gumbel_generator.sample()
        out_matrix_c = 1.0*(torch.sign(out_matrix-1/2)+1)/2
        err = torch.sum(torch.abs(out_matrix_c * get_offdiag(sz) - obj_matrix * get_offdiag(sz))) # 计算时不考虑对角元素
        err = err.cpu() if use_cuda else err
        errs.append(err.data.numpy())

        # 计算tpr fpr
        tpr, fpr = calc_tpr_fpr(obj_matrix, out_matrix_c)
        tprs.append(tpr)
        fprs.append(fpr)

    err_net = np.mean(errs)
    tpr_score = np.mean(tprs)
    fpr_score = np.mean(fprs)
    # print(fprs)

    return err_net, tpr_score, fpr_score


def skip_diag_strided(A):
    m = A.shape[0]
    strided = np.lib.stride_tricks.as_strided
    s0,s1 = A.strides
    return strided(A.ravel()[1:], shape=(m-1,m), strides=(s0+s1,s1)).reshape(m,-1)


def calc_tpr_fpr(matrix, matrix_pred):
    matrix = matrix.to('cpu').data.numpy()
    matrix_pred = matrix_pred.to('cpu').data.numpy()

    # 去对角元素
    matrix = skip_diag_strided(matrix)
    matrix_pred = skip_diag_strided(matrix_pred)

    # 计算指标
    tn, fp, fn, tp = confusion_matrix(matrix.astype(int).reshape(-1),
                                      matrix_pred.astype(int).reshape(-1)).ravel()
    # print(tn, fp, fn, tp)

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    return tpr, fpr