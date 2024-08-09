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
from utils import *

from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from models import ECC_Transformer

des_dir       =  '/content/ecct_data/Result/'
model_direct     = '/content/ecct_data/Model/'
ds_dir        =  '/content/ecct_data/Dataset/'

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.device_count()
print(device)

if __name__ == "__main__":
    class Code:
        pass
    code = Code()
    code_files = os.listdir(ds_dir)
    for tmp in code_files:

      if 'checkpoints' not in tmp and 'Wimax_LDPC' not in tmp:
        print(tmp.split('_'))
        code.n = int(tmp.split('_')[1][1:])
        code.k = int(tmp.split('_')[-1][1:].split('.')[0])
        code.code_type = tmp.split('_')[0]



class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        #self.allzero_m = torch.zeros((self.code.k)).long() if zero_cw else None
        self.allzero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            #m = self.allzero_m
            x = self.allzero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        emb = torch.cat([magnitude, syndrome], -1)
        return x.float(), z.float(), y.float(), emb.float()


##################################################################
def attn_map(attn, subfix):
  attn = attn.cpu().detach().numpy()
  plt.close('all')
  ax_1 = plt.gca()
  im_1 = ax_1.matshow(attn)
  plt.colorbar(im_1)
  plt.savefig(f'{des_dir}ECCT_{subfix}.png', bbox_inches = 'tight')


##################################################################

def train(model, device, args, train_loader, optimizer, epoch, LR, flag):
    model.train()
    #scaler = torch.cuda.amp.GradScaler()
    cum_loss = cum_ber = cum_fer = cum_samples = cum_mloss = cum_darm_loss = 0
    grad_bank = {}
    t = time.time()
    for batch_idx, (x, z, y, magsynd) in enumerate(train_loader):
        #z_mul = (y * bin_to_sign(x))
        #with torch.autocast(device_type='cuda', dtype=torch.float16):
        z_pred = model(magsynd.to(device))
        loss, x_pred = model.loss(-z_pred, y.to(device), x.to(device))

        #if flag:
        #  with torch.no_grad():
        #    for n, p in model.named_parameters():
        #      grad_bank[n] = p.grad

        model.zero_grad(set_to_none=True)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        #cum_darm_loss += darmloss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    print(f'Training epoch {epoch}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e}, BER={cum_ber / cum_samples:.2e}, Train Time {time.time() - t :.2f}s')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')

    if flag:
      np.save(f'{des_dir}_Baseline_{epoch}_{args.code_type}_{args.code_n}_{args.code_k}_mode_grad.npy', grad_bank)

    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    t = time.time()
    save_str = f'Base{len(model.decoder.layers)}L_bs{args.batch_size}_e{args.epochs}_{args.code_type}_{args.code_n}_{args.code_k}_NonSys'
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            t1 = time.time()
            while True:
                (m, x, z, y, magsynd) = next(iter(test_loader))
                z_pred = model(magsynd.to(device))
                loss, x_pred = model.loss(-z_pred, y.to(device), x.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                ebno = EbNo_range_test[ii]
                if min_FER > 0:
                    if ebno < 4:
                      if cum_count > 1e4 and test_fer > min_FER:
                        break
                    else:
                      if cum_count >= 1e8:
                        #print(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                        break
                      elif cum_count > 1e5 and test_fer > min_FER:
                        #print(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                        break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)

            #for p in range(x.size(0)):
            #  z_hard = sign_to_bin(torch.sign(y[p] * bin_to_sign(x[p])))
            #  if (z_hard.sum() < 5):
            idx = 0
            #    break

            #GET Y
            attn_map(y[0].unsqueeze(-1), f'{save_str}_RX_EbNo{ii}')
            attn_map(sign_to_bin(torch.sign(y[0])).unsqueeze(-1), f'{save_str}_RX_HardDecision_EbNo{ii+1}')

            #GET NN Input
            emb = torch.cat((magnitude, syndrome), -1)
            attn_map(emb[0].unsqueeze(-1), f'{save_str}_ECCT_Input_EbNo{ii+1}')

            #GET Pred
            attn_map(x_pred[0].unsqueeze(-1), f'{save_str}_Prediction_EbNo{ii+1}')

            for j in range(args.N_dec):
              attn_map(model.decoder.layers[j].self_attn.attn[idx,:,:,:].mean(0), f'{save_str}_AttnW_meanheads_L{j+1}_EbNo{ii+1}')

            print(f'Test EbN0={EbNo_range_test[ii]}, FER={test_loss_fer_list[-1]:.2e}, BER={test_loss_ber_list[-1]:.2e}, Total samples = {cum_count}, Test time = {time.time() - t1:.2f}')
        ###
        logging.info('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        logging.info('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        logging.info('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
        logging.info('Test -ln(BER) ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, -np.log(elem)) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_list, test_loss_ber_list, test_loss_fer_list

##################################################################
##################################################################
##################################################################


def main(args):#, model):
    code = args.code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #################################
    model = ECC_Transformer(args, dropout=0)
    if args.loadmodel:
        model = torch.load(f'{des_dir}BCH__Code_n_511_k_466__23_04_2024_23_38_08_Base_6L_NonSys/best_model')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs - 0, eta_min=5e-7) #1000

    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    #################################
    EbNo_range_test = range(1, 8)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(ECC_Dataset(code, std_train, len=args.batch_size * 1000, zero_cw=True), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers, pin_memory=True)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers, pin_memory=True) for ii in range(len(std_test))]
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1): #1001
        if epoch in [100, 300, 500, 1000]:
          getgrad = True
        else:
          getgrad = False
        loss, ber, fer = train(model, device, args, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0], flag=getgrad)
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
        if epoch in [1, args.epochs] :
            test(model, device, test_dataloader_list, EbNo_range_test)

##################################################################################################################
##################################################################################################################
##################################################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loadmodel', default=False)

    # Code args
    parser.add_argument('--code_type', type=str, default='PolarCode',
                        choices=['BCH', 'PolarCode', 'LDPC', 'CCSDS', 'MACKAY', 'Wimax'])
    parser.add_argument('--code_k', type=int, default=768)
    parser.add_argument('--code_n', type=int, default=1024)
    parser.add_argument('--standardize', default=True)# #action='store_true')


    # model args
    parser.add_argument('--modeltype', type=str, default="UECCT", choices=['ECCT', 'UECCT', 'VUECCT'])
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument("-f", required=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    print(args.standardize)

    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    print(args.code_type)
    G, H = Get_Generator_and_Parity(code,standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ####################################################################
    model_dir = os.path.join(des_dir,
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) + '__' + datetime.now().strftime(
                                 f"%d_%m_%Y_%H_%M_%S_Base_{args.N_dec}L_Sys"))
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    #model = ECC_Transformer(args, dropout=0).to(device)
    main(args)#, model)
