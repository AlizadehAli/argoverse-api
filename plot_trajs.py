from data_loader import ArgoDataset
from loss_functions import multiMSE, multiNLL
from dumb_predictor import DumbPredictor
from torch.utils.data import DataLoader
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

val_data_dir = 'forecasting_sample/data/'
val_lane_data_dir = 'forecasting_sample_lanes/data/'
load_file = ''
batch_size = 1
n_test = 10

val_set = ArgoDataset(val_data_dir, val_lane_data_dir, random_rotation=True, random_translation=False)

train_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True,
                               num_workers=0, collate_fn=val_set.collate_fn)
net = DumbPredictor()
if torch.cuda.is_available():
    net = net.cuda()
    if load_file != '':
        print('Loading ', load_file)
        net.load_state_dict(torch.load(load_file + '.tar'))
else:
    if load_file != '':
        print('Loading ', load_file)
        net.load_state_dict(torch.load(load_file + '.tar', map_location='cpu'))

def plot_arrows(x, y, color, alpha=None, ax=None):
    x_pos = x[:-1]
    y_pos = y[:-1]

    vec_x = x[1:]-x_pos
    vec_y = y[1:]-y_pos

    if ax is None:
        plt.quiver(x_pos, y_pos, vec_x, vec_y, scale_units='xy', angles='xy', scale=1,
                   width=0.004, color=color, alpha=alpha)
    else:
        ax.quiver(x_pos, y_pos, vec_x, vec_y, scale_units='xy', angles='xy', scale=1,
                   width=0.004, color=color, alpha=alpha)

def plot_lanes(x, y, color="grey", alpha=None, ax=None):
    if ax is None:
        plt.plot(x, y, alpha=alpha, linewidth=1, zorder=0)
    else:
        ax.plot(x, y, alpha=alpha, linewidth=1, zorder=0)

for test in range(n_test):
    it_train = iter(train_data_loader)

    len_train = len(it_train)
    avg_loss = 0
    avg_mse = 0
    for batch_num in range(len_train):
        x, y, mask_x, mask_y, lanes = next(it_train)

        x_np = x.detach().numpy()
        y_np = y.detach().numpy()
        lanes_np = lanes.detach().numpy()

        mask_x_np = mask_x.detach().numpy()
        mask_y_np = mask_y.detach().numpy()

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            mask_x = mask_x.cuda()
            mask_y = mask_y.cuda()
            lanes = lanes.cuda()

        print("Number of Lanes: ", lanes.shape[2])
        print("Number of Vehicles: ", x.shape[2])

        y_pred = net(x, lanes, 30, mask_x, False)

        y_pred_np = y_pred.detach().numpy()

        loss = multiNLL(y_pred, y, mask_y)
        mse = multiMSE(y_pred, y, mask_y)

        print('Loss ', loss)
        print('Mse ', mse)

        # Create figure and axes
        fig, ax = plt.subplots(1)
        for b in range(x.shape[1]):

            for v in range(x.shape[2]):
                not_zero_fut = np.nonzero(mask_y_np[:, b, v])
                not_zero_hist = np.nonzero(mask_x_np[:, b, v])
                fut_x = y_np[not_zero_fut[0], b, v, 0]
                fut_y = y_np[not_zero_fut[0], b, v, 1]
                hist_x = x_np[not_zero_hist[0], b, v, 0]
                hist_y = x_np[not_zero_hist[0], b, v, 1]
                if len(hist_x) < 2 or len(hist_y) < 2:
                    continue
                # plt.plot(hist_x, hist_y, '4-', color='blue')
                plot_arrows(hist_x, hist_y, color='blue', ax=ax)

                height = 1.7
                width = 3.95
                angle = np.arctan2(hist_y[-2]-hist_y[-1], hist_x[-2] - hist_x[-1])
                c2 = np.cos(angle) / 2
                s2 = np.sin(angle) / 2
                center_x = hist_x[-1] - (c2 * width - s2 * height)
                center_y = hist_y[-1] - (s2 * width + c2 * height)
                angle = angle*180/np.pi
                if v == 0:
                    color = "red"
                elif v == 1:
                    color = "green"
                else:
                    color = "gray"

                ax.add_artist(
                    patches.Rectangle((center_x, center_y), width, height, angle=angle,
                                      edgecolor=color, facecolor=color, alpha=0.5, fill=True)
                )
                mean_p = np.prod(y_pred_np[not_zero_fut[0], b, v, :, 5], axis=0) ** (1 / x.shape[0])
                mean_p = np.log(1+mean_p*1e3)
                mean_p = mean_p / np.max(mean_p)
                for g in range(y_pred_np.shape[3]):
                    if not_zero_hist[0][-1]:
                        pred_x = y_pred_np[:, b, v, g, 0]
                        pred_y = y_pred_np[:, b, v, g, 1]
                        # plt.plot(pred_x, pred_y, '4-', color='red', alpha=mean_p[g])
                        plot_arrows(pred_x, pred_y, color='red', alpha=mean_p[g], ax=ax)

                # plt.plot(fut_x, fut_y, '4-', color='green')
                plot_arrows(fut_x, fut_y, color='green', ax=ax)

                ax.set_aspect(1)
            for g in range(lanes_np.shape[2]):
                plot_lanes(lanes_np[:, b, g, 0], lanes_np[:, b, g, 1])
            plt.show()

