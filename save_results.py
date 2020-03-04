import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle

from get_model import get_model
from fusion_dataset_loader import FusionDataset
from torch.utils.data import DataLoader

from loss_functions import multiMSE, multiNLL, multiADE,\
    multiFDE, minADE, minFDE, missRate, multiNLLBest, multiDPP

# dataset_path = "forecasting_sample/dataset2/"
log_dir = 'runs'
# dataset_path = 'val/dataset2/data+lanes.pickle'
# dataset = pickle.load(open(dataset_path, 'rb'))
load_model_name = 'model_geometricAttention_best_122_fusion8_bis'


dataset_path = '../data_fusion/SUV_TF_60kkm/test_sequenced_data.tar'
dataset = FusionDataset(dataset_path, random_rotation=False)
batch_size = 64
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=dataset.collate_fn)
from torch.utils.data import DataLoader
output_path = log_dir + '/' + load_model_name + '/'
open_loop = False
len_hist = 50
min_num_obs = 3
len_lane = 40
len_pred = 75
n_points_slope = 20


def get_net(load_model_name):
    return get_model('runs', load_model_name, load_model_name, False, True, len_hist, len_lane, len_pred)


net = get_net(load_model_name)


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
    traj_past, lanes, angle = normalize_data(traj_past, lanes)
    # if traj_past.shape[1] == 1:
    #     traj_past = np.concatenate((traj_past, traj_past), axis=1)
    #     mask_past = np.concatenate((mask_past, np.zeros_like(mask_past)), axis=1)
    traj_fut = traj_past[len_hist:]
    traj_past = traj_past[:len_hist]
    mask_fut = mask_past[len_hist:]
    mask_past = mask_past[:len_hist]
    if open_loop:
        traj_past = traj_past[1:] - traj_past[:-1]
    traj_past = torch.from_numpy(traj_past.astype('float32'))
    traj_fut = torch.from_numpy(traj_fut.astype('float32'))
    mask_past = torch.from_numpy(mask_past.astype('bool'))
    mask_fut = torch.from_numpy(mask_fut.astype('bool'))
    lanes = torch.from_numpy(lanes.astype('float32'))
    mask_lanes = torch.from_numpy(mask_lanes.astype('bool'))
    if torch.cuda.is_available():
        traj_past = traj_past.cuda()
        traj_fut = traj_fut.cuda()
        mask_past = mask_past.cuda()
        mask_fut = mask_fut.cuda()
        lanes = lanes.cuda()
        mask_lanes = mask_lanes.cuda()

    return traj_past.unsqueeze(1), traj_fut.unsqueeze(1), mask_past.unsqueeze(1), mask_fut.unsqueeze(1),\
           lanes.unsqueeze(1), mask_lanes.unsqueeze(1), mean_pos, angle

net.set_training(False)
counter = 0
output_all = []
hist_test = []
mask_test = []
fut_test = []
pred_test = []
lines_test = []
path_list = dataset.dataset['path']
len_dataset = len(dataset)
# for idx, data in dataset.items():
for idx, data in enumerate(dataLoader):
    print('\r' + str(counter) + '/' + str(len_dataset), end="")
    # traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, mean_pos, angle = get_torch_data(data)
    if torch.cuda.is_available():
        data = [data_i.cuda() for data_i in data]
    traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes = data

    if open_loop:
        current_pos = traj_past[-1:]
    else:
        current_pos = None
    pred_fut = net(traj_past, mask_past,
                   lanes, mask_lanes, len_pred, init_pos=current_pos)
    pred_fut = pred_fut[-len_pred:]
    if open_loop:
        pred_fut[:, :, :, :, :2] = \
            torch.cumsum(pred_fut[:, :, :, :, :2], dim=0) + current_pos[:, :, :1, :].unsqueeze(3)
    pred_fut[:, :, :, :, 5] = torch.mean(pred_fut[:, :, :, :, 5], dim=0, keepdim=True)
    current_prediction = sort_predictions(pred_fut.cpu().detach().clone().numpy())
    current_traj_fut = traj_fut.cpu().detach().clone().numpy()
    current_mask_fut = mask_fut.cpu().detach().clone().numpy()
    lines = lanes.cpu().detach().numpy()
    # nll = multiNLL(current_prediction, current_traj_fut, current_mask_fut).item()
    # mse = multiMSE(current_prediction, current_traj_fut, current_mask_fut).item()
    # ade = multiADE(current_prediction, current_traj_fut, current_mask_fut).item()
    # min_ade = minADE(current_prediction, current_traj_fut, current_mask_fut).item()
    # min_fde = minFDE(current_prediction, current_traj_fut, current_mask_fut).item()
    # miss_rate = missRate(current_prediction, current_traj_fut, current_mask_fut).item()

    # output_all.append({'idx': idx, 'nll': nll, 'mse': mse, 'ade': ade, 'min_ade': min_ade,
    #                    'min_fde': min_fde, 'miss_rate': miss_rate})
    lines_test.append(lines)
    hist_test.append(traj_past.cpu().detach().numpy())
    mask_test.append(current_mask_fut)
    fut_test.append(current_traj_fut)
    pred_test.append(current_prediction)

    counter += batch_size


with open(output_path + load_model_name + '.pickle', 'wb') as handle:
    pickle.dump({'hist':hist_test, 'mask':mask_test, 'fut':fut_test, 'pred':pred_test, 'path':path_list, 'lines': lines_test}, handle, protocol=pickle.HIGHEST_PROTOCOL)
# np.savez_compressed(output_path + 'error_data.npz', hist=hist_test, mask=mask_test, fut=fut_test,
#                     pred=pred_test, path=path_list)

# with open(output_path + 'error_data.pickle', 'wb') as handle:
#     pickle.dump(output_all, handle,
#                 protocol=pickle.HIGHEST_PROTOCOL)