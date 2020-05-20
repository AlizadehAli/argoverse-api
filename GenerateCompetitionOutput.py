from argoverse.evaluation.competition_util import generate_forecasting_h5
from dataset_loader import ArgoDataset
from torch.utils.data import DataLoader
from loss_functions import xytheta2xy

# from attention_predictor import AttentionPredictor
import matplotlib.pyplot as plt

import numpy as np
import torch
import pickle
import h5py
import os

from get_model import get_model


##set root_dir to the correct path to your dataset folder
root_dir = 'test_obs/dataset2/data+lanes.pickle'
# root_dir = 'forecasting_sample/dataset/'
# root_dir = 'competition_files/input_data.pickle'
dataset = pickle.load(open(root_dir, 'rb'))

print('Total number of sequences:', len(dataset))

len_hist = 20
len_lane = 70
len_pred = 30
n_points_slope = 20
batch_size = 32
open_loop = False
load_model_name = 'model_sumAttention_122_loopy_5'
net = get_model('runs', load_model_name, load_model_name, False, len_hist, len_lane, len_pred)
net = net.cuda()
# net.set_training(True)

test_set = ArgoDataset('test_obs/dataset2/', normalize=True)
test_set.time_len = 20
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=test_set.collate_fn)

def scene_rotation(coor, angle):
    rot_matrix = np.zeros((2, 2))
    c = np.cos(angle)
    s = np.sin(angle)
    rot_matrix[0, 0] = c
    rot_matrix[0, 1] = -s
    rot_matrix[1, 0] = s
    rot_matrix[1, 1] = c
    coor = np.matmul(rot_matrix, np.expand_dims(coor, axis=-1))
    return coor.squeeze(-1)


def sort_predictions(pred_fut):
    pred_fut[:, :, :, :, 5] = np.mean(pred_fut[:, :, :, :, 5], axis=0, keepdims=True)
    flat_pred_test = pred_fut.reshape([-1, 6, 6])
    flat_argsort_p = np.argsort(flat_pred_test[:, :, 5], axis=1)[:, ::-1]
    flat_pred_test_sorted_p = flat_pred_test.copy()
    for i in range(6):
        flat_pred_test_sorted_p[:, i, :] = flat_pred_test[np.arange(flat_pred_test.shape[0]), flat_argsort_p[:, i]]
    return flat_pred_test_sorted_p.reshape(
        [pred_fut.shape[0], pred_fut.shape[1], pred_fut.shape[2], pred_fut.shape[3], pred_fut.shape[4]])


def normalize_data(traj, lanes):
    X = traj[len_hist - n_points_slope:len_hist, 0, 0]
    Y = traj[len_hist - n_points_slope:len_hist, 0, 1]
    mdX = np.mean(X[1:] - X[:-1])
    mdY = np.mean(Y[1:] - Y[:-1])
    angle = -np.arctan2(mdY, mdX)
    angle += np.pi / 4

    traj = scene_rotation(traj, angle)
    lanes = scene_rotation(lanes, angle)
    return traj, lanes, angle

def get_torch_data(data):
    traj_past = data['traj']
    mask_past = data['mask_traj']
    lanes = data['lanes']
    mask_lanes = data['mask_lanes']
    len_pred = data['len_pred']
    mean_pos = data['mean_pos']
    if traj_past.shape[1] == 1:
        traj_past = np.concatenate((traj_past, traj_past), axis=1)
        mask_past = np.concatenate((mask_past, np.zeros_like(mask_past)), axis=1)
    traj_past, lanes, angle = normalize_data(traj_past, lanes)
    traj_fut = traj_past[len_hist:]
    traj_past = traj_past[:len_hist]
    if open_loop:
        traj_past = traj_past[1:] - traj_past[:-1]
    traj_past = torch.from_numpy(traj_past.astype('float32'))
    mask_past = torch.from_numpy(mask_past.astype('bool'))
    lanes = torch.from_numpy(lanes.astype('float32'))
    mask_lanes = torch.from_numpy(mask_lanes.astype('bool'))
    if torch.cuda.is_available():
        traj_past = traj_past.cuda()
        mask_past = mask_past.cuda()
        lanes = lanes.cuda()
        mask_lanes = mask_lanes.cuda()
    return traj_past.unsqueeze(1), traj_fut, mask_past.unsqueeze(1), lanes.unsqueeze(1), mask_lanes.unsqueeze(1), mean_pos, angle

counter = 0
# output_all = {}
# for idx, data in dataset.items():
#     traj_past, traj_fut, mask_past, lanes, mask_lanes, mean_pos, angle = get_torch_data(data)
#     print('\r'+str(counter)+'/'+str(len(dataset)), end="")
#     if open_loop:
#         current_pos = traj_past[-1:]
#     else:
#         current_pos = None
#     pred_fut = net(traj_past, mask_past,
#                    lanes, mask_lanes, len_pred, init_pos=current_pos)
#     if open_loop:
#         pred_fut[:, :, :, :, :2] = \
#         torch.cumsum(pred_fut[:, :, :, :, :2], dim=0) + current_pos[:, :, :1, :].unsqueeze(3)
#     pred_fut = pred_fut.detach().cpu().numpy()
#
#     pred_fut = sort_predictions(pred_fut)
#
#     # flat_pred_test = pred_fut.reshape([-1, 6, 6])
#     # flat_argsort_p = np.argsort(flat_pred_test[:, :, 5], axis=1)
#     # flat_pred_test_sorted_p = flat_pred_test.copy()
#     # for i in range(6):
#     #     flat_pred_test_sorted_p[:, i, :] = flat_pred_test[np.arange(flat_pred_test.shape[0]), flat_argsort_p[:, i]]
#     # pred_test_max_p = flat_pred_test_sorted_p.reshape(
#     #     [pred_fut.shape[0], pred_fut.shape[1], pred_fut.shape[2], pred_fut.shape[3], pred_fut.shape[4]])
#
#     output_all[idx] = scene_rotation(pred_fut[:, 0, 0, :, :2], -angle).transpose((1, 0, 2)) + mean_pos
#     counter += 1
n_pred = 6
future_frames = 30
output_all2 = np.zeros([len(dataset)*n_pred*future_frames, 3], dtype=np.float32)
id_list = test_set.idx_list
for batch, data in enumerate(test_loader):
    batch_past, batch_fut, mask_past, mask_fut, batch_lanes, mask_lanes, angle, mean_pos = data
    if torch.cuda.is_available():
        batch_past = batch_past.cuda()
        mask_past = mask_past.cuda()
        batch_lanes = batch_lanes.cuda()
        mask_lanes = mask_lanes.cuda()
    print('\r' + str(counter) + '/' + str(len(dataset)), end="")
    pred_fut = net(batch_past, mask_past, batch_lanes, mask_lanes, len_pred)
    pred_fut = xytheta2xy(pred_fut[:, :, 0:1, :, :], 4)
    pred_fut = pred_fut.detach().cpu().numpy()
    pred_fut = sort_predictions(pred_fut)
    batch_size = batch_past.shape[1]
    for i in range(batch_size):
        pred = scene_rotation(pred_fut[:, i, 0, :, :2], -angle[i]).transpose((1, 0, 2)) + mean_pos[i]
        output_all2[(counter + i)*n_pred*future_frames:(counter + i + 1)*n_pred*future_frames, 0] = id_list[counter + i]
        output_all2[(counter + i)*n_pred*future_frames:(counter + i + 1)*n_pred*future_frames, 1:] = pred.reshape((n_pred*future_frames, 2))
    counter += batch_size

output_path = 'competition_files/'
hf = h5py.File(os.path.join(output_path, load_model_name + ".h5"), "w")
#output_all2 = output_all2.tolist()
hf.create_dataset("argoverse_forecasting", data=output_all2, compression="gzip", compression_opts=9)
hf.close()
# generate_forecasting_h5(output_all2, output_path, load_model_name)
