from data_loader import ArgoDataset
from loss_functions import multiMSE, multiNLL, multiDPP
from dumb_predictor import DumbPredictor
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
from logger import Logger
from shutil import copyfile
from ranger import Ranger

train_data_dir = 'train/data/'
val_data_dir = 'val/data/'
log_dir = 'logs/'
files_to_copy = ['dumb_predictor.py', 'train_model.py', 'loss_functions.py', 'data_loader.py']
out_model_name = 'model_road_attentionx2_sep(6)_12'
load_name = ''
agent_only = False
batch_size = 16
n_epoch = 30
lr = 0.001

logger = Logger(log_dir+out_model_name)
for file_name in files_to_copy:
    copyfile(file_name, log_dir + out_model_name + '/' + file_name)

train_set = ArgoDataset(train_data_dir, random_rotation=True, random_translation=False)
val_set = ArgoDataset(val_data_dir, random_rotation=True, random_translation=False)

train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=0, collate_fn=train_set.collate_fn)

val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=val_set.collate_fn)

net = DumbPredictor()
if torch.cuda.is_available():
    net = net.cuda()
    if load_name != '':
        print('Loading ', load_name)
        net.load_state_dict(torch.load('./logs/' + load_name + '/' + load_name + '.tar'))
else:
    if load_name != '':
        print('Loading ', load_name)
        net.load_state_dict(torch.load('./logs/' + load_name + '/' + load_name + '.tar', map_location='cpu'))

optimizer = Ranger(net.parameters(), lr=lr)

for epoch in range(n_epoch):
    it_train = iter(train_data_loader)
    len_train = len(it_train)
    avg_loss = 0
    avg_nll = 0
    avg_mse = 0
    avg_diversity = 0
    for batch_num in range(len_train):
        x, y, mask_x, mask_y, lanes, mask_lanes = next(it_train)

        x_np = x.detach().numpy()
        y_np = y.detach().numpy()

        mask_x_np = mask_x.detach().numpy()
        mask_y_np = mask_y.detach().numpy()

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            mask_x = mask_x.cuda()
            mask_y = mask_y.cuda()
            lanes = lanes.cuda()

        y_pred = net(x, lanes, len_pred=30, mask=mask_x, is_pretrain=False)

        if agent_only:
            y_pred = y_pred[:, :, 0:1, :, :]
            y = y[:, :, 0:1, :]

        nll = multiNLL(y_pred, y, mask_y)
        mse = multiMSE(y_pred, y, mask_y)
        diversity = multiDPP(y_pred, mask_y)

        loss = nll



        avg_loss += loss.detach()
        avg_mse += mse.detach()
        avg_nll += nll.detach()
        avg_diversity += diversity.detach()

        if batch_num%100 == 99:
            torch.save(net.state_dict(), log_dir + out_model_name + '/' + out_model_name + '.tar')
            avg_loss = avg_loss.item()
            avg_mse = avg_mse.item()
            avg_nll = avg_nll.item()
            avg_diversity = avg_diversity.item()
            print("Epoch no:", epoch, "| progress(%):",
                  format(batch_num / len_train * 100, '0.2f'),
                  '| loss:', format(avg_loss / 100, '0.4f'),
                  '| nll:', format(avg_nll / 100, '0.4f'),
                  '| mse:', format(avg_mse / 100, '0.4f'),
                  '| diversity:', format(avg_diversity / 100, '0.4f'))
            info = {'nll': avg_nll / 100, 'mse': avg_mse / 100}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, int((epoch * len_train + batch_num) / 100))
            avg_loss = 0
            avg_mse = 0
            avg_nll = 0
            avg_diversity = 0

    print('######## Validation ########')
    it_val = iter(val_data_loader)
    len_val = len(it_val)
    avg_loss = 0
    avg_mse = 0
    for batch_num in range(len_val):
        x, y, mask_x, mask_y, lanes = next(it_val)

        x_np = x.detach().numpy()
        y_np = y.detach().numpy()

        mask_x_np = mask_x.detach().numpy()
        mask_y_np = mask_y.detach().numpy()

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            mask_x = mask_x.cuda()
            mask_y = mask_y.cuda()
            lanes = lanes.cuda()

        y_pred = net(x, lanes, 30, mask_x, False)

        if agent_only:
            y_pred = y_pred[:, :, 0:1, :, :]
            y = y[:, :, 0:1, :]

        loss = multiNLL(y_pred, y, mask_y)
        mse = multiMSE(y_pred, y, mask_y)

        avg_loss += loss.detach()
        avg_mse += mse.detach()

    torch.save(net.state_dict(), log_dir + out_model_name + '/' + out_model_name + '.tar')
    avg_loss = avg_loss.item()
    avg_mse = avg_mse.item()
    print("Epoch no:", epoch,
          '| loss:', format(avg_loss / len_val, '0.4f'),
          '| mse:', format(avg_mse / len_val, '0.4f'))
    info = {'val_nll': avg_loss / len_val, 'val_mse': avg_mse / len_val}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, int((epoch + 1) * len_train / 100))
    print('############################')
