from lane_follow_predictor import LaneFollowPredictor

from dataset_loader import ArgoDataset
from torch.utils.data import DataLoader
from torch.optim import Adam

from loss_functions import multiNLL, multiMSE, multiFDE, missRate, multiADE, minADE, minFDE

import torch

from ranger import Ranger
# from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

def isnan(x):
    return x != x

class AttentionPredictionTrainer:
    def __init__(self, train_dataset_dir, val_dataset_dir, out_model_name,
                 batch_size=8, n_epoch=16, lr=0.0003,
                 log_dir='runs/', load_model_name=None):
        self.out_model_name = out_model_name
        self.log_dir = log_dir
        self.len_pred = 30
        self.len_hist = 20
        self.len_lane = 70
        self.scale_factor = 5
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.logger = SummaryWriter(log_dir + out_model_name)
        train_set = ArgoDataset(train_dataset_dir, normalize=True, random_translation=True, random_rotation=True)
        val_set = ArgoDataset(val_dataset_dir, normalize=True)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                        num_workers=0, collate_fn=train_set.collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                        num_workers=0, collate_fn=train_set.collate_fn)
        files_to_copy = ['train_lane_follow_predictor.py', 'lane_follow_predictor.py', 'dataset_loader.py',
                         'attention.py', 'loss_functions.py']
        for file_name in files_to_copy:
            copyfile(file_name, log_dir + out_model_name + '/' + file_name)

        self.is_cuda = torch.cuda.is_available()

        self.net = LaneFollowPredictor(self.len_hist, self.len_lane, self.len_pred)

        if load_model_name is not None:
            print('Loading ', load_model_name)
            self.net.load_state_dict(
                torch.load(log_dir + load_model_name + '/' + load_model_name + '.tar', map_location='cpu'))

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.n_data = len(self.train_loader)
        self.n_val_data = len(self.val_loader)

        self.avg_nll = 0
        self.avg_mse = 0
        self.avg_ade = 0
        self.avg_min_ade = 0
        self.avg_fde = 0
        self.avg_min_fde = 0
        self.avg_miss_rate = 0
        self.metric_itemized = False

        self.current_traj_past  = None
        self.current_traj_fut   = None
        self.current_mask_past  = None
        self.current_mask_fut   = None
        self.current_lanes      = None
        self.current_mask_lanes = None


    def _reset_avg_metrics(self):
        self.avg_nll = 0
        self.avg_mse = 0
        self.avg_ade = 0
        self.avg_fde = 0
        self.avg_min_ade = 0
        self.avg_min_fde = 0
        self.avg_miss_rate = 0
        self.metric_itemized = False

    def _get_current_batch(self, data):
        self.current_traj_past, self.current_traj_fut, self.current_mask_past,\
        self.current_mask_fut, self.current_lanes, self.current_mask_lanes = data
        if self.is_cuda:
            self.current_traj_past  = self.current_traj_past.cuda()
            self.current_traj_fut   = self.current_traj_fut.cuda()
            self.current_mask_past  = self.current_mask_past.cuda()
            self.current_mask_fut   = self.current_mask_fut.cuda()
            self.current_lanes      = self.current_lanes.cuda()
            self.current_mask_lanes = self.current_mask_lanes.cuda()

    def _get_loss(self, only_agent=False):
        if only_agent:
            current_prediction = self.current_prediction[:, :, 0:1, :].clone()
            current_traj_fut = self.current_traj_fut[:, :, 0:1, :].clone()
            current_mask_fut = self.current_mask_fut[:, :, 0:1].clone()
            loss = multiMSE(current_prediction,
                             current_traj_fut,
                             current_mask_fut, p_ind=2)
        else:
            loss = multiMSE(self.current_prediction, self.current_traj_fut, self.current_mask_fut, p_ind=2)
        assert not isnan(loss)
        return loss

    def _update_avg_metrics(self):
        if self.metric_itemized:
            print("Average metrics implicitely reset.")
            self._reset_avg_metrics()
        current_prediction = self.current_prediction[:, :, 0:1, :].detach().clone().rename(None)
        current_prediction = current_prediction*self.scale_factor
        current_traj_fut = self.current_traj_fut[:, :, 0:1, :].detach().clone().rename(None)*self.scale_factor
        current_mask_fut = self.current_mask_fut[:, :, 0:1].detach().clone().rename(None)
        # self.avg_nll = self.avg_nll + multiNLL(current_prediction,
        #                                        current_traj_fut,
        #                                        current_mask_fut)
        self.avg_mse = self.avg_mse + multiMSE(current_prediction,
                                                current_traj_fut,
                                                current_mask_fut, p_ind=2)
        self.avg_ade = self.avg_ade + multiADE(current_prediction,
                                                current_traj_fut,
                                                current_mask_fut, p_ind=2)
        self.avg_min_ade = self.avg_min_ade + minADE(current_prediction,
                                                     current_traj_fut,
                                                     current_mask_fut)
        self.avg_fde = self.avg_fde + multiFDE(current_prediction,
                                               current_traj_fut,
                                               current_mask_fut, p_ind=2)
        self.avg_min_fde = self.avg_min_fde + minFDE(current_prediction,
                                                     current_traj_fut,
                                                     current_mask_fut)
        self.avg_miss_rate = self.avg_miss_rate + missRate(current_prediction,
                                                           current_traj_fut,
                                                           current_mask_fut)

    def _itemize_avg_metrics(self):
        # self.avg_nll = self.avg_nll.item()
        self.avg_mse = self.avg_mse.item()
        self.avg_ade = self.avg_ade.item()
        self.avg_fde = self.avg_fde.item()
        self.avg_min_ade = self.avg_min_ade.item()
        self.avg_min_fde = self.avg_min_fde.item()
        self.avg_miss_rate = self.avg_miss_rate.item()
        self.metric_itemized = True

    def _print_metrics(self, epoch, batch, is_val=False):
        if not self.metric_itemized:
            self._itemize_avg_metrics()
        if is_val:
            prefix = 'val_'
            denom = self.n_val_data
            progress = epoch/self.n_epoch * 100
        else:
            prefix = ''
            denom = 100
            progress = batch / self.n_data * 100
        print("Epoch no:", epoch, "| progress(%):",
              format(progress, '0.2f'),
              # '| ' + prefix +'nll:', format(self.avg_nll / denom, '0.2f'),
              '| ' + prefix +'mse:', format(self.avg_mse / denom, '0.2f'),
              '| ' + prefix +'ade:', format(self.avg_ade / denom, '0.2f'),
              '| ' + prefix +'min_ade:', format(self.avg_min_ade / denom, '0.2f'),
              '| ' + prefix +'fde:', format(self.avg_fde / denom, '0.2f'),
              '| ' + prefix +'min_fde:', format(self.avg_min_fde / denom, '0.2f'),
              '| ' + prefix +'miss_rate:', format(self.avg_miss_rate / denom, '0.2f')
              )

    def _log(self, batch_num, is_val=False):
        if is_val:
            prefix = 'val_'
            denom = self.n_val_data
            name = 'Validation'
        else:
            prefix = ''
            denom = 100
            name = 'Training'

        info = {#prefix+'nll': self.avg_nll / denom,
                prefix+'mse': self.avg_mse / denom,
                prefix+'ade': self.avg_ade / denom,
                prefix+'fde': self.avg_fde / denom,
                prefix+'min_ade': self.avg_min_ade / denom,
                prefix+'min_fde': self.avg_min_fde / denom,
                prefix+'miss_rate': self.avg_miss_rate / denom
                }
        for key, value in info.items():
            self.logger.add_scalar(name + '/'+ key, value, batch_num)

    def _save_model(self):
        torch.save(self.net.state_dict(), self.log_dir +
                   self.out_model_name + '/' + self.out_model_name + '.tar')

    def _predict(self):
        self.current_prediction = self.net(self.current_traj_past, self.current_mask_past,
                                           self.current_lanes, self.current_mask_lanes, self.len_pred)
        return self.current_prediction

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.optimizer.step()

    def _validation(self, epoch):
        print('                  ################### Validation ####################')
        self._reset_avg_metrics()
        for batch, data in enumerate(self.val_loader):
            self._get_current_batch(data)
            self._predict()
            self._update_avg_metrics()
        self._print_metrics(epoch, 0, is_val=True)
        self._save_model()
        self._log(int((epoch + 1)*self.n_data / 100), is_val=True)
        print('                  ##################################################')

    def train(self):
        self._reset_avg_metrics()
        for epoch in range(self.n_epoch):
            for batch, data in enumerate(self.train_loader):
                self._get_current_batch(data)
                self._predict()
                self._update_avg_metrics()
                loss = self._get_loss(only_agent=False)
                self._step(loss)
                if batch%100 == 99:
                    self._save_model()
                    self._print_metrics(epoch, batch)
                    self._log(int((epoch * self.n_data + batch) / 100))
                    self._reset_avg_metrics()
            self._validation(epoch)


trainer = AttentionPredictionTrainer(
    # train_dataset_dir='forecasting_sample/dataset2/',
    # val_dataset_dir='forecasting_sample/dataset2/',
    # out_model_name='test',
    train_dataset_dir='train/dataset2/',
    val_dataset_dir='val/dataset2/',
    out_model_name='model_laneFollow_97',
    batch_size=16,
    n_epoch=300,
    lr=0.0003,
    log_dir='runs/',
    load_model_name=None
    )

trainer.train()

