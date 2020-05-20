from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics

import numpy as np
import torch
import  pickle

from get_model import get_model


##set root_dir to the correct path to your dataset folder
# root_dir = 'test_obs/data/'
root_dir = 'forecasting_sample/data/'
# root_dir = 'train/data'
# root_dir = 'val/data'

max_num_lanes = 30
max_size_lane = 10
dist_max = 30
len_hist = 20
len_lane = 70
len_pred = 30
print_metrics = False
save_inputs = True
# load_model_name = 'model_lane_52'
# net = get_model('runs', load_model_name, load_model_name, len_hist, len_lane, len_pred)
afl = ArgoverseForecastingLoader(root_dir)
print('Total number of sequences:', len(afl))
avm = ArgoverseMap()

counter = 0
output_all = {}
input_all = {}
truth_all = {}
city_names = {}

for data in afl:
    data_frame = data.seq_df
    tracks = data_frame.groupby('TRACK_ID')
    time_list = sorted(data_frame["TIMESTAMP"].unique())
    len_time = len(time_list)
    all_lanes = []
    all_mask_lanes = []
    all_traj = []
    all_mask_traj = []
    mask_traj_temp = np.ones([len_time])
    traj_temp = np.zeros([len_time, 2])
    for track_id, track_data in tracks:
        mask_traj_temp.fill(1)
        traj_temp.fill(0)
        city_name = data.city
        x = track_data["X"]
        y = track_data["Y"]

        traj = np.column_stack((x, y))
        time = track_data["TIMESTAMP"].values
        if traj.shape[0] == len_time:
            traj_temp = traj
        else:
            args = np.searchsorted(time_list, time)
            mask_traj_temp.fill(0)
            mask_traj_temp[args] = 1
            traj_temp[args, :] = traj

        object_type = track_data["OBJECT_TYPE"].values[0]
        if object_type == "AGENT":
            ############# Get lanes corresponding to agent #############
            obs_traj = traj_temp[:len_hist][mask_traj_temp[:len_hist].astype('bool')]
            if len(obs_traj) > 3:
                # lanes = avm.get_candidate_centerlines_for_traj(obs_traj, city_name)
                lanes = avm.find_local_lane_centerlines(obs_traj[-1, 0], obs_traj[-1, 1], city_name, query_search_range_manhattan=dist_max)
                if lanes.ndim == 3:
                    lanes = lanes[:, :, :2]
                else:
                    lanes = []
            else:
                lanes = []

            lane_temp = np.zeros([max_num_lanes, max_size_lane, 2])
            mask_lane_temp = np.zeros([max_num_lanes, max_size_lane])
            for ind, lane in enumerate(lanes):
                if ind >= max_num_lanes - 1:
                    break
                len_lane = lane.shape[0]
                if len_lane < max_size_lane:
                    lane_temp[ind, :len_lane, :] = lane
                    mask_lane_temp[ind, :len_lane] = 1
                else:
                    lane_temp[ind, :, :] = lane[-max_size_lane:, :]
                    mask_lane_temp[ind, :] = 1
            all_traj.insert(0, traj_temp.copy())
            all_mask_traj.insert(0, mask_traj_temp.copy())
            all_lanes = lane_temp.copy()
            all_mask_lanes = mask_lane_temp.copy()
        elif object_type == "AV":
            pos_insert = min(len(all_traj), 1)
            all_traj.insert(pos_insert, traj_temp.copy())
            all_mask_traj.insert(pos_insert, mask_traj_temp.copy())
            # all_lanes.insert(pos_insert, lane_temp.copy())
            # all_mask_lanes.insert(pos_insert, mask_lane_temp.copy())
        else:
            if np.any(traj_temp[:len_hist] != 0):
                all_traj.append(traj_temp.copy())
                all_mask_traj.append(mask_traj_temp.copy())
                # all_lanes.append(lane_temp.copy())
                # all_mask_lanes.append(mask_lane_temp.copy())

    traj = np.stack(all_traj).transpose([1, 0, 2])
    mask_traj = np.stack(all_mask_traj).transpose([1, 0])
    lanes = np.stack(all_lanes).transpose([1, 0, 2])
    mask_lanes = np.stack(all_mask_lanes).transpose([1, 0])

    ##############  Relative data and filter out ###################

    mean_pos = traj[19, 0, :]

    rel_traj = (traj - mean_pos) * (mask_traj[:, :, None])
    rel_lane = (lanes - mean_pos) * (mask_lanes[:, :, None])
    # mask vehicle allways further than dist_max or not seen (keep only vehicle seen at less than dist_max)
    rel_dist = np.square(rel_traj[:, :, 0]) + np.square(rel_traj[:, :, 1])
    mask_dist = np.any(np.logical_and(rel_dist != 0, rel_dist < dist_max*dist_max), axis=0)
    # mask vehicles that where not observed enough
    mask_obs = np.all(mask_traj[17:20, :] != 0, axis=0)
    mask_veh = np.logical_and(mask_dist, mask_obs)
    assert mask_veh[0]
    traj = rel_traj[:, mask_veh, :]
    mask_traj = mask_traj[:, mask_veh]
    seq_id = int(data.current_seq.name[:-4])

    if save_inputs:
        input_all[seq_id] = {'traj': traj, 'mask_traj': mask_traj,
                             'lanes': rel_lane, 'mask_lanes': mask_lanes, 'len_pred': len_pred, 'mean_pos': mean_pos}

    # ############### To torch ####################
    #
    # traj = torch.from_numpy(traj.astype('float32'))
    # mask_traj = torch.from_numpy(mask_traj.astype('bool'))
    # lanes = torch.from_numpy(lanes.astype('float32'))
    # mask_lanes = torch.from_numpy(mask_lanes.astype('bool'))
    #
    # if torch.cuda.is_available():
    #     traj = traj.cuda()
    #     mask_traj = mask_traj.cuda()
    #     lanes = lanes.cuda()
    #     mask_lanes = mask_lanes.cuda()
    #
    # traj = traj.unsqueeze(1)
    # mask_traj = mask_traj.unsqueeze(1)
    # lanes = lanes.unsqueeze(1)
    # mask_lanes = mask_lanes.unsqueeze(1)
    #
    print('\r'+str(counter)+'/'+str(len(afl)), end="")
    # pred_fut = net(traj[:len_hist], mask_traj[:len_hist],
    #                lanes, mask_lanes, len_pred)
    # pred_fut = pred_fut.detach().cpu().numpy()
    # pred_fut[:, :, :, :, 5].sort(axis=3)
    # output_all[seq_id] = pred_fut[:, 0, 0, :, :2].transpose((1, 0, 2)) + mean_pos
    #
    if print_metrics:
        truth_all[seq_id] = traj[len_hist:, 0, 0, :]
        city_names[seq_id] = data.city

    counter += 1

if print_metrics:
    compute_forecasting_metrics(output_all, truth_all, city_names, max_n_guesses=6, horizon=len_pred, miss_threshold=2)
output_path = 'forecasting_sample/dataset2/'

# generate_forecasting_h5(output_all, output_path)

if save_inputs:
    with open(output_path + 'data+lanes.pickle', 'wb') as handle:
        pickle.dump(input_all, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
