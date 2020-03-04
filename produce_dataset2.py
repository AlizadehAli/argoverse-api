from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.evaluation.eval_forecasting import compute_forecasting_metrics

import numpy as np
import pickle

import multiprocessing


##set root_dir to the correct path to your dataset folder
# root_dir = 'val/data/'
# target_dir = 'val/dataset/'
# root_dir = 'forecasting_sample/data/'
# target_dir = 'forecasting_sample/dataset'
# root_dir = 'val/data'
root_dir = 'test_obs/data/'
target_dir = 'test_obs/dataset/'

pool_size = 10
max_num_lanes = 5
max_size_lane = 70
dist_max = 30
len_hist = 20
len_lane = 70
len_pred = 30
truth_available = True
afl = ArgoverseForecastingLoader(root_dir)
print('Total number of sequences:', len(afl))
avm = ArgoverseMap()

def save_dataset_part(indices, name):
    counter = 0
    traj_dataset = []
    lanes_dataset = []
    mask_traj_dataset = []
    mask_lanes_dataset = []
    seq_id_dataset = []
    # for data in afl:
    for i in indices:
        data = afl[i]
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

            ############# Get lanes corresponding to traj #############
            obs_traj = traj_temp[:len_hist][mask_traj_temp[:len_hist].astype('bool')]
            if len(obs_traj) > 3:
                lanes = avm.get_candidate_centerlines_for_traj(obs_traj, city_name)
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

            object_type = track_data["OBJECT_TYPE"].values[0]
            if object_type == "AGENT":
                all_traj.insert(0, traj_temp.copy())
                all_mask_traj.insert(0, mask_traj_temp.copy())
                all_lanes.insert(0, lane_temp.copy())
                all_mask_lanes.insert(0, mask_lane_temp.copy())
            elif object_type == "AV":
                pos_insert = min(len(all_traj), 1)
                all_traj.insert(pos_insert, traj_temp.copy())
                all_mask_traj.insert(pos_insert, mask_traj_temp.copy())
                all_lanes.insert(pos_insert, lane_temp.copy())
                all_mask_lanes.insert(pos_insert, mask_lane_temp.copy())
            else:
                if np.any(traj_temp[:len_hist] != 0):
                    all_traj.append(traj_temp.copy())
                    all_mask_traj.append(mask_traj_temp.copy())
                    all_lanes.append(lane_temp.copy())
                    all_mask_lanes.append(mask_lane_temp.copy())

        traj = np.stack(all_traj).transpose([1, 0, 2])
        mask_traj = np.stack(all_mask_traj).transpose([1, 0])
        lanes = np.stack(all_lanes).transpose([2, 0, 1, 3])
        mask_lanes = np.stack(all_mask_lanes).transpose([2, 0, 1])

        ##############  Relative data and filter out ###################

        mean_pos = traj[19, 0, :]

        rel_traj = (traj - mean_pos) * (mask_traj[:, :, None])
        rel_lane = (lanes - mean_pos) * (mask_lanes[:, :, :, None])
        # mask vehicle allways further than dist_max or not seen (keep only vehicle seen at less than dist_max)
        rel_dist = np.square(rel_traj[:, :, 0]) + np.square(rel_traj[:, :, 1])
        mask_dist = np.any(np.logical_and(rel_dist != 0, rel_dist < dist_max*dist_max), axis=0)
        # mask vehicles that where not observed enough
        mask_obs = np.all(mask_traj[17:20, :] != 0, axis=0)
        mask_veh = np.logical_and(mask_dist, mask_obs)
        assert mask_veh[0]
        traj = rel_traj[:, mask_veh, :]
        lanes = rel_lane[:, mask_veh, :, :]
        mask_lanes = mask_lanes[:, mask_veh, :]
        mask_traj = mask_traj[:, mask_veh]

        seq_id = int(data.current_seq.name[:-4])
        traj_dataset.append(traj)
        mask_traj_dataset.append(mask_traj)
        lanes_dataset.append(lanes)
        mask_lanes_dataset.append(mask_lanes)
        seq_id_dataset.append(seq_id)

        counter += 1
    print('\r' + str(i) + '/' + str(len(indices)), end="")
    print('Len', len(traj_dataset), 'items', len(indices))
    with open(target_dir + '/' + name + '.pickle', 'wb') as handle:
        pickle.dump({'traj': traj_dataset, 'mask_traj': mask_traj_dataset,
                     'lanes': lanes_dataset, 'mask_lanes': mask_lanes_dataset, 'seq_id': seq_id_dataset}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

def map_produce_dataset(idx):
    print('process', idx)
    size = len(afl) // (pool_size - 1)
    beg = idx * size
    end = min(beg + size, len(afl))
    print("beg ", beg, 'end', end)
    list = np.arange(beg, end)
    name = str(idx)
    save_dataset_part(list, name)


pool = multiprocessing.Pool(pool_size)
pool.map(map_produce_dataset, np.arange(pool_size))
