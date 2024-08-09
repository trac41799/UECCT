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
from submodules import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BaselineEncoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)


class UEncoder(nn.Module):
    def __init__(self, args, top, mid2, bot, N): #mid1,
        super().__init__()
        self.code = args.code
        self.top_layers = clones(top, N//3)
        self.mid2_layers = clones(mid2, N//3)
        self.bot_layers = clones(bot, 1)
        #bot_layers = clones(feed_forward, 4)
        #self.layers =  mid_layers + top_layers #+ bot_layers
        self.norm = LayerNorm(top.size)
        if N > 1:
            self.norm2 = LayerNorm(bot.size)
        #    self.norm2 = LayerNorm(top.size + 32)
        #self.sublayer = clones(SublayerConnection(layer.size, dropout = 0), N)
        #self.bnk1 = nn.Linear(bot.size, bot.size)
        self.bnk1 = nn.Linear(bot.size, bot.size + 32)

        self.rescale_1 = nn.Linear(args.d_model, top.size)
        self.rescale_2 = nn.Linear(top.size, mid2.size)
        self.rescale_4 = nn.Linear(mid2.size, bot.size)
        self.rescale_5 = nn.Linear(bot.size + 32, bot.size)
        self.rescale_6 = nn.Linear(bot.size, mid2.size)
        self.rescale_8 = nn.Linear(mid2.size, top.size)

        self.bias =  2.0 #torch.nn.Parameter(torch.Tensor([3.0]))
        self.scaler1 = 2.0 #torch.nn.Parameter(torch.Tensor([10.0]))
        self.scaler2 = 2.0 #torch.nn.Parameter(torch.Tensor([2.0]))

        self.top_skip_map = [0]
        self.top_map = [0]
        self.top_mul = [0]
        self.top_post_product = [0]

        self.bot_skip_map = [0]
        self.bot_map = [0]
        self.bot_mul = [0]
        self.bot_post_product = [0]

        self.mid_skip_map = [0]
        self.mid_map = [0]
        self.mid_mul = [0]
        self.mid_post_product = [0]


    def F_filter(self, x, skip):
        x = x*skip
        return x

    def forward(self, x, mask, flag = False):
        #for idx, layer in enumerate(self.layers, start=1):
        x = self.rescale_1(x)
        x = self.top_layers[0](x, mask) # self.rescale_1(x)
        x = self.top_layers[1](x, mask)
        top_skip = x #self.F_filter(x)

        x = self.rescale_2(x)
        x = self.mid2_layers[0](x, mask) #
        x = self.mid2_layers[1](x, mask)
        mid2_skip = x #self.F_filter(x)

        x = self.rescale_4(x)
        x = self.bot_layers[0](x, mask) #
        #x = self.bot_layers[0](x, mask)
        bot = x #self.F_filter(x)

        x = self.norm2(x)
        x = F.gelu(self.bnk1(x))

        x = self.rescale_5(x)
        if flag == True:
          self.bot_skip_map = bot
          self.bot_map = x
        x = self.F_filter(x, bot)
        if flag == True:
          self.bot_post_product = x
        #x = self.bot_layers[0](x, mask)
        x = self.bot_layers[0](x, mask)

        x = self.rescale_6(x)
        if flag == True:
          self.mid_skip_map = mid2_skip
          self.mid_map = x
        x = self.F_filter(x, mid2_skip)
        #self.mid_mul = x
        #x = F.normalize(x, dim = (1,2))
        if flag == True:
          self.mid_post_product = x

        x = self.mid2_layers[1](x, mask)
        x = self.mid2_layers[0](x, mask)

        x = self.rescale_8(x)
        if flag == True:
          self.top_skip_map = top_skip
          self.top_map = x
        x = self.F_filter(x, top_skip)
        #x = F.normalize(x, dim = (1,2))
        if flag == True:
          self.top_post_product = x

        x = self.top_layers[1](x, mask)
        x = self.top_layers[0](x, mask)

        return self.norm(x)


class VUEncoder(nn.Module):
    def __init__(self, args, top, mid2, bot, N): #mid1,
        super().__init__()
        self.args = args
        self.top_size = top.size
        self.mid2_size = mid2.size
        self.bot_size = bot.size
        self.src_embed1 = torch.nn.Parameter(torch.empty(
            (args.code.n + args.code.pc_matrix.size(0), args.d_model)))

        self.sampler = Sampler(bot.size) #bot.size)#args.code.n)# + args.code.pc_matrix.size(0)) #
        self.top_layers = clones(top, N//3)
        #self.mid1_layers = clones(mid1, N//3)
        self.mid2_layers = clones(mid2, N//3)
        self.bot_layers = clones(bot, 1)
        #bot_layers = clones(feed_forward, 4)
        #self.layers =  mid_layers + top_layers #+ bot_layers
        self.norm = LayerNorm(top.size)
        if N > 1:
            #self.norm1 = LayerNorm(top.size)
            #self.norm2 = LayerNorm(mid2.size)
            self.norm3 = LayerNorm(bot.size)#args.code.n + args.code.pc_matrix.size(0)) #
        #    self.norm3 = LayerNorm(bot.size)
        #self.sublayer = clones(SublayerConnection(layer.size, dropout = 0), N)
        self.bnk1 = nn.Linear(bot.size, bot.size)
        self.pretop = None
        self.premid = None
        self.prebot = None
        self.latent = None
        self.postbot = None
        self.postmid = None
        self.posttop = None
        self.final = None
        self.flag = False

    def forward(self, x, mask):
        #z, z_mean, z_logvar = self.sampler(z)
        #z = z.unsqueeze(-1) * self.src_embed2.unsqueeze(0)
        x = self.src_embed1.unsqueeze(0) * x
        #x = self.rescale_1(x)
        x = self.top_layers[0](x, x, self.darm, mask)
        x = self.top_layers[1](x, x, self.darm, mask)
        top_skip = x #self.upscale1(x) # x.std(dim = -1)
        #top_skip = F.pad(x, (0, self.mid2_size - self.top_size), "constant", 0)

        #x = self.rescale_2(x)
        x = self.mid2_layers[0](x, x, self.darm, mask)
        x = self.mid2_layers[1](x, x, self.darm, mask)
        mid_skip = x #self.upscale2(x) #x.std(dim = -1)
        #mid_skip = x #F.pad(x, (0, self.bot_size - self.mid2_size), "constant", 0)

        #x = self.rescale_4(x)
        x = self.bot_layers[0](x, x, self.darm, mask)
        #x = self.bot_layers[1](x, x, mask)
        # = self.bot_layers[0](x, x, mask)
        #x = self.norm3(x)
        bot = x #.std(dim = -1) #F.tanh(self.pool3(x)).squeeze(-1) #

        if self.flag:
          self.pretop = top_skip
          self.premid = mid_skip
          self.prebot = bot

        z = bot + top_skip + mid_skip # torch.cat([bot, top_skip, mid_skip],-1).sum(dim = -1) #bot + top_skip + mid_skip #
        #z = self.z_resize(z)
        z = self.norm3(z)

        z, z_mean, z_logvar = self.sampler(z)
        #z = self.z_inverse(z).unsqueeze(-1)#torch.cat([torch.abs(z), z_syndrome], -1).unsqueeze(-1) #self.z_inverse(z).unsqueeze(-1)
        #z_emb = z.mean(dim = -1, keepdim = True)#.unsqueeze(-1)
        #z = z.mean(dim = -1, keepdim = True)
        #z = z * self.src_embed_top.unsqueeze(0)#self.downscale1(z) #
        #z_mid = z * self.src_embed_mid.unsqueeze(0)#self.downscale2(z) #
        #z_bot = z * self.src_embed_bot.unsqueeze(0)

        #x = self.norm3(x)
        x = F.gelu(self.bnk1(x))

        #x = self.rescale_5(x) #torch.cat((x_skip, bot),-1))
        x_mul = x + z
        if self.flag:
          self.latent = z
          self.postbot = x_mul
        x = self.bot_layers[0](x_mul, x_mul, self.darm, mask, True)
        #x = self.bot_layers[0](x, x, mask, True)
        #x = self.rescale_6(x)#torch.cat((mid2_skip, x),-1))
        #x = x + z_mid
        if self.flag:
          self.postmid = x + z
        x = self.mid2_layers[1](x, x + z, self.darm, mask, True)
        x = self.mid2_layers[0](x, x + z, self.darm, mask, True)

        if self.flag:
          self.posttop = x + z
        #x = self.rescale_8(x)#torch.cat((top_skip, x),-1))
        x = self.top_layers[1](x, x + z, self.darm, mask, True)
        x = self.top_layers[0](x, x + z, self.darm, mask, True)
        if self.flag:
          self.final = x
        return self.norm(x), z_mean, z_logvar


############################################################


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        self.args = args
        code = args.code
        c = copy.deepcopy
        #attn = MultiHeadedAttention(args.h, args.d_model)
        #ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)
        size1 = (args.d_model//2)
        size2 = (3*args.d_model//4)
        size3 = (5*args.d_model//4)
        print(f'L1: {size1}, L2: {size2}, L3: {size3}')

        self.src_embed = torch.nn.Parameter(torch.empty(
            (code.n + code.pc_matrix.size(0), args.d_model)))
        if args.modeltype == 'ECCT':
            self.decoder = BaselineEncoder(EncoderLayer(
                args, dropout), args.N_dec)
        elif args.modeltype == 'UECCT':
            self.decoder = UEncoder(args, EncoderLayer(
                                size1, dropout, args, True),
                               EncoderLayer(
                                size2, dropout, args, True),
                               EncoderLayer(
                                size3, dropout, args, True),
                               args.N_dec)
        elif args.modeltype == 'VUECCT':
            self.decoder = VUEncoder(args, EncoderLayer(
                                       args.d_model, dropout, args, True),
                                   EncoderLayer(
                                       args.d_model, dropout, args, True),
                                   EncoderLayer(
                                       args.d_model, dropout, args, True),
                                   args.N_dec)
        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear((args.d_model//2), 1)])
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)

        self.get_mask(code)
        logging.info(f'Mask:\n {self.src_mask}')
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, invec, device, flag = False):

        emb = self.src_embed.unsqueeze(0).to(device) * invec.unsqueeze(-1)
        if self.args.modeltype == 'VUECCT':
            emb, z_mean, z_logvar = self.decoder(emb, self.src_mask, flag)
            return self.out_fc(self.oned_final_embed(emb).squeeze(-1)), z_mean, z_logvar

        else:
            emb = self.decoder(emb, self.src_mask, flag)
            return self.out_fc(self.oned_final_embed(emb).squeeze(-1))


    def loss(self, z_pred, x, y, z_mean = None, z_logvar = None):
        z2 = y*x
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z2)))
        if self.args.modeltype == 'VUECCT':
            kl_target = torch.normal(mean=torch.zeros(z_mean.size()).to(device),
                                     std=torch.std(y, dim=-1, keepdim=True).unsqueeze(-1))
            # torch.randn(np.shape(z_mean)).to(device) * torch.std(y, dim = -1, keepdim = True).unsqueeze(-1)
            kl_pred = torch.normal(mean=z_mean, std=torch.sqrt(torch.exp(z_logvar)))
            kl_loss = -F.kl_div(kl_pred, kl_target, reduction='none').mean()
            loss = loss + kl_loss
        
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask_size = code.n + code.pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk:
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[code.n + ii, jj] += 1
                            mask[jj, code.n + ii] += 1
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(code)
        self.register_buffer('src_mask', src_mask)
