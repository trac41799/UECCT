from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from matplotlib import pyplot as plt

from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from utils import *
from models import clones

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sampler(nn.Module):
    def __init__(self, size): #mid1,
        super(Sampler, self).__init__()
        self.size = size
        self.sampling = nn.Linear(size, size*2)
        #self.z_logvar_layer = nn.Linear(size, size)


    def forward(self, x):
        z = self.sampling(x)
        z_mean   = z[:,:,:self.size]#self.z_mean_layer(x)
        z_logvar = z[:,:,self.size:]#self.z_logvar_layer(x)
        eps = torch.randn(np.shape(z_logvar))
        samples = z_mean + torch.exp(0.5 * z_logvar) * eps.to(device)
        return samples, z_mean, z_logvar


class PrenormResConnection(nn.Module):
    def __init__(self, size, dropout):
        super(PrenormResConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, dropout, args, inp = False):
        super(EncoderLayer, self).__init__()
        self.mhsa = MHA(args.h, size)
        self.ffn = PFFN(size, size*4, dropout)
        self.sublayer = clones(PrenormResConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.mhsa(x, x, x, mask)) #[0]
        return self.sublayer[1](x, self.ffn)


class MHA(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MHA, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.dynmask = 0
        self.err = torch.Tensor([0.0])

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)
        #, self.err, self.dynmask
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores / math.sqrt(d_k), dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PFFN, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))