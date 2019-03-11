#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:03:55 2018

@author: xinruyue
"""
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from utils.util import gumbel_softmax

use_cuda = torch.cuda.is_available()
np.random.seed(2050)
torch.manual_seed(2050)
if use_cuda:
    torch.cuda.manual_seed(2050)

# Network Generator
# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator(nn.Module):
    def __init__(self, sz = 10, temp = 10, temp_drop_frac = 0.9999):
        super(Gumbel_Generator, self).__init__()
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        #gen_matrix 为邻接矩阵的概率
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
    def drop_temperature(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac
    def sample(self, hard=False):
        # 采样——得到一个临近矩阵
        self.logp = self.gen_matrix.view(-1, 2)
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        if use_cuda:
            out = out.cuda()
        out_matrix = out[:,0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix
    def get_temperature(self):
        return self.temperature
    def get_cross_entropy(self, obj_matrix):
        # 计算与目标矩阵的距离
        logps = F.softmax(self.gen_matrix, 2)
        logps = torch.log(logps[:,:,0] + 1e-10) * obj_matrix + torch.log(logps[:,:,1] + 1e-10) * (1 - obj_matrix)
        result = - torch.sum(logps)
        result = result.cpu() if use_cuda else result
        return result.data.numpy()
    def get_entropy(self):
        logps = F.softmax(self.gen_matrix, 2)
        result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
        result = result.cpu() if use_cuda else result
        return(- result.data.numpy())
    def randomization(self, fraction):
        # 将gen_matrix重新随机初始化，fraction为重置比特的比例
        sz = self.gen_matrix.size()[0]
        numbers = int(fraction * sz * sz)
        original = self.gen_matrix.cpu().data.numpy()
        
        for i in range(numbers):
            ii = np.random.choice(range(sz), (2, 1))
            z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
            self.gen_matrix.data[ii[0], ii[1], :] = z


#Graph Network （Dynamics Learning）
class GumbelGraphNetwork(nn.Module):
    def __init__(self, input_size, hidden_size = 128):
        super(GumbelGraphNetwork, self).__init__()
        self.edge1 = torch.nn.Linear(2 * input_size, hidden_size)
        self.edge2edge = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(input_size + hidden_size, input_size)
        
    def forward(self, x, adj,skip_conn=0):
        out = x
        innode = out.unsqueeze(1).repeat(1, adj.size()[1], 1, 1)
        outnode = innode.transpose(1, 2)
        node2edge = F.relu(self.edge1(torch.cat((innode,outnode), 3)))
        edge2edge = F.relu(self.edge2edge(node2edge))
        adjs = adj.view(adj.size()[0], adj.size()[1], adj.size()[2], 1)
        adjs = adjs.repeat(1, 1, 1, edge2edge.size()[3])
        
        edges = adjs * edge2edge
        out = torch.sum(edges, 1)
        out = F.relu(self.node2node(out))
        out = F.relu(self.node2node2(out))
        out = torch.cat((x, out), dim = -1)
        out = self.output(out)
        if skip_conn == 1:
            out = out + x
        return out

# nn
class GumbelGraphNetworkClf(nn.Module):
    def __init__(self, input_size, hidden_size = 256):
        super(GumbelGraphNetworkClf, self).__init__()
        self.edge1 = torch.nn.Linear(2 * input_size, hidden_size)
        self.edge2edge = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(input_size+hidden_size, input_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.test1 = torch.nn.Linear(hidden_size, hidden_size)
        self.test2 = torch.nn.Linear(hidden_size, hidden_size)
        self.test3 = torch.nn.Linear(hidden_size, hidden_size)
    def forward(self, x, adj):
        out = x
        innode = out.unsqueeze(1).repeat(1, adj.size()[1], 1, 1)
        outnode = innode.transpose(1, 2)
        node2edge = F.relu(self.edge1(torch.cat((innode,outnode), 3)))
        edge2edge = F.relu(self.edge2edge(node2edge))
        adjs = adj.view(adj.size()[0], adj.size()[1], adj.size()[2], 1)
        adjs = adjs.repeat(1, 1, 1, edge2edge.size()[3])

        edges = adjs * edge2edge

        out = torch.sum(edges, 1)
        out = F.relu(self.node2node(out))
        out = F.relu(self.node2node2(out))
        out = torch.cat((x, out), dim = -1)
        out = self.output(out)
        out = self.logsoftmax(out)
        return out
