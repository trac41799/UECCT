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



def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def Read_pc_matrixrix_alist(fileName):
    with open(fileName, 'r') as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(
            lines[0].rstrip('\n'), dtype=int, sep=' ')
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip('\n'), dtype=int, sep=' ')
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H
#############################################
def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
        idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]

def get_standard_form(pc_matrix_):
    pc_matrix = pc_matrix_.copy().astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist()+[min(pc_matrix.shape) - 1,next_col]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist()+[next_col, ii]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)
#############################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_Generator_and_Parity(code, standard_form = False):
    n, k = code.n, code.k
    path_pc_mat = os.path.join(ds_dir, f'{code.code_type}_N{str(n)}_K{str(k)}')
    if code.code_type in ['PolarCode', 'BCH']:
        ParityMatrix = np.loadtxt(path_pc_mat+'.txt')
    elif code.code_type in ['CCSDS', 'LDPC', 'MACKAY', 'Wimax']:
        ParityMatrix = Read_pc_matrixrix_alist(path_pc_mat+'.txt')
    else:
        raise Exception(f'Wrong code {code.code_type}')
    if standard_form and code.code_type not in ['CCSDS', 'LDPC', 'MACKAY','Wimax']:
        ParityMatrix = get_standard_form(ParityMatrix).astype(int)
        GeneratorMatrix = np.concatenate([np.mod(-ParityMatrix[:, min(ParityMatrix.shape):].transpose(),2),np.eye(k)],1).astype(int)
    else:
        GeneratorMatrix = get_generator(ParityMatrix)
    assert np.all(np.mod((np.matmul(GeneratorMatrix, ParityMatrix.transpose())), 2) == 0) and np.sum(GeneratorMatrix) > 0
    return GeneratorMatrix.astype(float), ParityMatrix.astype(float)

