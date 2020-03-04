from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap

from data_loader import ArgoDataset

import pandas as pd
import numpy as np
import pickle

import multiprocessing

from argoverse.utils.centerline_utils import (
    get_nt_distance,
    remove_overlapping_lane_seq,
)

base_name = 'test_obs'
# base_name = 'forecasting_sample'

dataset_dir = '/home/jean/Tensorflow/Argoverse/'+base_name+'/dataset/'

root_dir = '/home/jean/Tensorflow/Argoverse/'+base_name+'/data/'
target_dir = '/home/jean/Tensorflow/Argoverse/'+base_name + '_lanes/data/'

afl = ArgoverseForecastingLoader(root_dir)
avm = ArgoverseMap()
obs_len = 20

pool_size = 10

def produce_lane_files(idx):
    print("Process ", idx)
    len_data = len(afl)
    size = len_data // (pool_size - 1)
    beg = idx*size
    end = min(beg + size, len_data)
    print("Loop between", beg, end)
    for ind in range(beg, end):
        data = afl[ind]
        seq_id = int(data.current_seq.name[:-4])
        data_frame = data.seq_df
        tracks = data_frame.groupby('TRACK_ID')
        all_lanes = []
        for track_id, track_data in tracks:
            city_name = data.city
            x = track_data["X"]
            y = track_data["Y"]
            traj = np.column_stack((x, y))[:obs_len, :]
            if len(traj) > 3:
                candidate_centerlines = avm.get_candidate_centerlines_for_traj(traj, city_name)
                for lane_id, lane in enumerate(candidate_centerlines):
                    direction = avm.get_lane_direction(lane[0], city_name)
                    for xy in lane:
                        all_lanes.append({"TRACK_ID": track_id,
                                          "SEQ_ID": seq_id,
                                          "LANE_ID": lane_id,
                                          "LANE_X": xy[0],
                                          "LANE_Y": xy[1],
                                          "DIRECTION": direction})
        lanes_df = pd.DataFrame(all_lanes)
        # data_frame = pd.merge(data_frame, lanes_df, on='TRACK_ID')
        lanes_df.to_csv(target_dir + data.current_seq.name)
        print("saved", target_dir+data.current_seq.name)

# produce_lane_files(0)
pool = multiprocessing.Pool(pool_size)
pool.map(produce_lane_files, np.arange(pool_size))

dataset = ArgoDataset(root_dir, target_dir, random_rotation=False, random_translation=False, get_id=True)
len_dataset = len(dataset)


def produce_dataset_temp(name, idx_list):
    traj_bunch = []
    mask_traj_bunch = []
    lanes_bunch = []
    mask_lanes_bunch = []
    track_id_bunch = []
    for idx in idx_list:
        traj, mask_traj, lanes, mask_lanes, id = dataset[idx]
        traj_bunch.append(traj)
        mask_traj_bunch.append(mask_traj)
        lanes_bunch.append(lanes)
        mask_lanes_bunch.append(mask_lanes)
        track_id_bunch.append(id)

    with open(dataset_dir + name + '_temp.pickle', 'wb') as handle:
        pickle.dump({'traj': traj_bunch, 'mask_traj': mask_traj_bunch,
                     'lanes': lanes_bunch, 'mask_lanes': mask_lanes_bunch, 'id': track_id_bunch}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def map_produce_dataset(idx):
    print('process', idx)
    size = len_dataset // (pool_size - 1)
    beg = idx * size
    end = min(beg + size, len_dataset)
    list = np.arange(beg, end)
    name = str(idx)
    produce_dataset_temp(name, list)


pool = multiprocessing.Pool(pool_size)
pool.map(map_produce_dataset, np.arange(pool_size))


def merge_dataset():
    with open(dataset_dir + 'dataset.pickle', "wb") as dataset_file:
        traj = []
        mask_traj = []
        lanes = []
        mask_lanes = []
        for i in range(pool_size):
            print(i)
            with open(dataset_dir + str(i) + '_temp.pickle', 'rb') as handle:
                data_temp = pickle.load(handle)
                traj = traj + data_temp['traj']
                mask_traj = mask_traj + data_temp['mask_traj']
                lanes = lanes + data_temp['lanes']
                mask_lanes = mask_lanes + data_temp['mask_lanes']
        pickle.dump({'traj':traj, 'mask_traj':mask_traj,
                     'lanes': lanes, 'mask_lanes': mask_lanes}, dataset_file,
                    protocol=pickle.HIGHEST_PROTOCOL)
# merge_dataset()