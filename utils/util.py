#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 10:59:56 2018

@author: xinruyue
"""
import sys
import torch
import numpy as np
import pickle as pkl

use_cuda = torch.cuda.is_available()
np.random.seed(2050)
torch.manual_seed(2050)
if use_cuda:
    torch.cuda.manual_seed(2050)

# Functions
def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    if use_cuda:
        gumbel = gumbel.cuda()
    return gumbel
def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax( y / temperature, dim = 1)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
      Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
      Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        #k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y
def get_offdiag(sz):
    ## 返回一个大小为sz的下对角线矩阵
    offdiag = torch.ones(sz, sz)
    for i in range(sz):
        offdiag[i, i] = 0
    if use_cuda:
        offdiag = offdiag.cuda()
    return offdiag       

def save_file(name, data, type):
    if type == 'pkl':
        with open(name + '.' + type, 'wb') as f:
            pkl.dump(data, f)
    if type == 'txt':
        with open(name + '.' + type, 'w') as f:
            for each in data:
                f.write(str(each))
                f.write('\n')

def read_file(name, type):
    if type == 'pkl':
        f = open(name + '.' + type, 'rb')
        data = pkl.load(f)
    if type == 'txt':
        f = open(name + '.' + type, 'r')
        data = []
        for line in f:
            line = line.strip('\n')
            line = float(line)
            data.append(line)
    if type == '':
        f = open(name, 'r')
        data = []
        for line in f:
            print(line)
            line = line.strip('\n')
            line = float(line)
            data.append(line)
    return data

