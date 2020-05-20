from attention_predictor import AttentionPredictor

from dataset_loader import ArgoDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from loss_functions import multiMSE, multiNLL, multiADE, multiFDE, minADE, minFDE, missRate, missLoss
from loss_functions import multiNLLBest, multiDPP, multiP
from loss_functions import xytheta_nll, xytheta2xy
from get_model import get_model

import torch

from ranger import Ranger
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile

def isnan(x):
    return x != x

class AttentionPredictionTrainer:
    def __init__(self, train_dataset_dir, val_dataset_dir, out_model_name,
                 batch_size=32, n_epoch=16, lr=0.0003,
                 log_dir='runs/', load_model_name=None, train_for_open_loop=False):
        self.train_for_open_loop = train_for_open_loop
        self.out_model_name = out_model_name
        self.log_dir = log_dir
        self.len_pred = 30
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.logger = SummaryWriter(log_dir + out_model_name)
        if train_for_open_loop:
            train_set = ArgoDataset(train_dataset_dir, normalize=False, random_translation=True, random_rotation=True)
            val_set = ArgoDataset(val_dataset_dir, normalize=False)
        else:
            train_set = ArgoDataset(train_dataset_dir, normalize=True, random_translation=True, random_rotation=True)
            val_set = ArgoDataset(val_dataset_dir, normalize=True)
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                        num_workers=0, collate_fn=train_set.collate_fn)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                        num_workers=0, collate_fn=train_set.collate_fn)
        files_to_copy = ['train_attention_predictor.py', 'attention_predictor.py', 'dataset_loader.py',
                         'loss_functions.py', 'get_model.py', 'attention.py']
        if load_model_name is not None:
            from_dir = log_dir + load_model_name + '/'
        else:
            from_dir = './'
        for file_name in files_to_copy:
            copyfile(from_dir+file_name, log_dir + out_model_name + '/' + file_name)

        self.is_cuda = torch.cuda.is_available()

        self.net = AttentionPredictor(separate_ego=False)

        if load_model_name is not None:
            print('Loading ', load_model_name)
            self.net = get_model(log_dir[:-1], load_model_name, load_model_name, False, 20, 70, self.len_pred)
            self.net.load_state_dict(
                torch.load(log_dir + load_model_name + '/' + load_model_name + '.tar', map_location='cpu'))

        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.n_data = len(self.train_loader)
        self.n_val_data = len(self.val_loader)

        self._reset_avg_metrics()

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
        self.avg_p = 0
        self.avg_min_ade = 0
        self.avg_min_fde = 0
        self.avg_miss_rate = 0
        self.avg_loss = 0
        self.metric_itemized = False

    def _get_current_batch(self, data):
        self.current_traj_past, self.current_traj_fut, self.current_mask_past,\
        self.current_mask_fut, self.current_lanes, self.current_mask_lanes, angle, mean_pos = data
        if self.is_cuda:
            self.current_traj_past  = self.current_traj_past.cuda()
            self.current_traj_fut   = self.current_traj_fut.cuda()
            self.current_mask_past  = self.current_mask_past.cuda()
            self.current_mask_fut   = self.current_mask_fut.cuda()
            self.current_lanes      = self.current_lanes.cuda()
            self.current_mask_lanes = self.current_mask_lanes.cuda()

    def _get_loss(self, only_agent=False):
        if only_agent:
            current_prediction = xytheta2xy(self.current_prediction[:, :, 0:1, :, :], 4).clone()
            current_traj_fut = self.current_traj_fut[:, :, 0:1, :2].clone()
            current_mask_fut = self.current_mask_fut[:, :, 0:1].clone()
            # loss = xytheta_nll(current_prediction,
            #                    current_traj_fut,
            #                    current_mask_fut, 4)
            loss = multiNLLBest(current_prediction, current_traj_fut, current_mask_fut, 4)# + multiP(current_prediction, current_mask_fut)
        else:
            current_prediction = xytheta2xy(self.current_prediction, 4).clone()
            current_traj_fut = self.current_traj_fut[:, :, :, :2].clone()
            current_mask_fut = self.current_mask_fut.clone()
            multip = multiP(current_prediction, current_mask_fut)
            loss = multiNLLBest(current_prediction, current_traj_fut, current_mask_fut, 4)\
                   + missLoss(current_prediction, current_traj_fut, current_mask_fut, 4)
            if not isnan(multip):
                loss += 0.0001*multip
        if isnan(loss):
            print("Loss value is Nan.")
            assert False
        return loss

    def _update_avg_metrics(self, loss=0):
        if self.metric_itemized:
            print("Average metrics implicitely reset.")
            self._reset_avg_metrics()
        current_prediction = xytheta2xy(self.current_prediction[:, :, 0:1, :, :], 4).detach().clone()
        current_traj_fut = self.current_traj_fut[:, :, 0:1, :2].detach().clone()
        current_mask_fut = self.current_mask_fut[:, :, 0:1].detach().clone()
        self.avg_nll = self.avg_nll + multiNLL(current_prediction,
                                               current_traj_fut,
                                               current_mask_fut)
        self.avg_mse = self.avg_mse + multiMSE(current_prediction,
                                               current_traj_fut,
                                               current_mask_fut)
        self.avg_ade = self.avg_ade + multiADE(current_prediction,
                                               current_traj_fut,
                                               current_mask_fut)
        self.avg_min_ade = self.avg_min_ade + minADE(current_prediction,
                                                     current_traj_fut,
                                                     current_mask_fut)
        self.avg_fde = self.avg_fde + multiFDE(current_prediction,
                                               current_traj_fut,
                                               current_mask_fut)
        self.avg_p = self.avg_p + multiP(current_prediction, current_mask_fut)
        self.avg_min_fde = self.avg_min_fde + minFDE(current_prediction,
                                                     current_traj_fut,
                                                     current_mask_fut)
        self.avg_miss_rate = self.avg_miss_rate + missRate(current_prediction,
                                                           current_traj_fut,
                                                           current_mask_fut)
        self.avg_loss = self.avg_loss + loss


    def _itemize_avg_metrics(self):
        self.avg_nll = self.avg_nll.item()
        self.avg_mse = self.avg_mse.item()
        self.avg_ade = self.avg_ade.item()
        self.avg_fde = self.avg_fde.item()
        self.avg_p = self.avg_p.item()
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
        to_print = "Epoch no:"+ str(epoch) + "| progress(%):"+\
              format(progress, '0.2f')+\
              '| ' + prefix +'nll: ' + format(self.avg_nll / denom, '0.2f')+\
              '| ' + prefix +'mse: ' + format(self.avg_mse / denom, '0.2f')+\
              '| ' + prefix +'ade: ' + format(self.avg_ade / denom, '0.2f')+\
              '| ' + prefix +'min_ade: ' + format(self.avg_min_ade / denom, '0.2f')+\
              '| ' + prefix +'fde: ' + format(self.avg_fde / denom, '0.2f')+\
              '| ' + prefix +'min_fde: ' + format(self.avg_min_fde / denom, '0.2f')+\
              '| ' + prefix +'self-similarity: ' + format(self.avg_p / denom, '0.2f')+\
              '| ' + prefix +'miss_rate: ' + format(self.avg_miss_rate / denom, '0.2f')
        if self.avg_loss != 0:
            to_print += '| ' + prefix +'loss:' + format(self.avg_loss / denom, '0.2f')
        print(to_print)


    def _log(self, batch_num, is_val=False):
        if is_val:
            prefix = 'val_'
            denom = self.n_val_data
            name = 'Validation'
        else:
            prefix = ''
            denom = 100
            name = 'Training'

        info = {prefix+'nll': self.avg_nll / denom,
                prefix+'mse': self.avg_mse / denom,
                prefix+'ade': self.avg_ade / denom,
                prefix+'fde': self.avg_fde / denom,
                prefix+'min_ade': self.avg_min_ade / denom,
                prefix+'min_fde': self.avg_min_fde / denom,
                prefix+'self-similarity': self.avg_p / denom,
                prefix+'miss_rate': self.avg_miss_rate / denom
                }
        for key, value in info.items():
            self.logger.add_scalar(name + '/'+ key, value, batch_num*self.batch_size/32)

    def _save_model(self):
        torch.save(self.net.state_dict(), self.log_dir +
                   self.out_model_name + '/' + self.out_model_name + '.tar')

    def _predict(self):
        if self.train_for_open_loop:
            current_pos = self.current_traj_past[-1:]
            self.current_traj_past = self.current_traj_past[1:] - self.current_traj_past[:-1]

            self.current_prediction = self.net(self.current_traj_past, self.current_mask_past,
                                               self.current_lanes, self.current_mask_lanes,
                                               self.len_pred, init_pos=current_pos)
            self.current_prediction[:, :, :, :, :2] = \
            torch.cumsum(self.current_prediction[:, :, :, :, :2], dim=0) + current_pos.unsqueeze(3)
        else:
            self.current_prediction = self.net(self.current_traj_past, self.current_mask_past,
                                               self.current_lanes, self.current_mask_lanes,
                                               self.len_pred)
        return self.current_prediction

    def _step(self, loss):
        if not isnan(loss):
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.1)
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

    def _get_net_size(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def train(self):
        self._reset_avg_metrics()
        n_weights = self._get_net_size()
        print("Training "+self.out_model_name+" that has "+str(n_weights)+" weights")
        for epoch in range(self.n_epoch):
            for batch, data in enumerate(self.train_loader):
                self._get_current_batch(data)
                self._predict()
                nll = self._get_loss(only_agent=False)
                self._update_avg_metrics(nll)
                self._step(nll)
                if batch%100 == 99:
                    self._save_model()
                    self._print_metrics(epoch, batch)
                    self._log(int((epoch * self.n_data + batch) / 100))
                    self._reset_avg_metrics()
            self._validation(epoch)

if __name__ == '__main__':
    trainer = AttentionPredictionTrainer(
        # train_dataset_dir='forecasting_sample/dataset2/',
        # val_dataset_dir='forecasting_sample/dataset2/',
        # out_model_name='test',
        train_dataset_dir='train/dataset2/',
        val_dataset_dir='val/dataset2/',
        out_model_name='model_sumAttention_122_loopy_15_6',
        batch_size=32,
        n_epoch=400,
        lr=0.0003,
        log_dir='runs/',
        load_model_name='model_sumAttention_122_loopy_15_5',
        train_for_open_loop=False
        )

    trainer.train()

