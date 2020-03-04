from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from lane_loader import DataLaneLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.map_representation.map_api import ArgoverseMap

from torch.utils.data import Dataset
import torch

import numpy as np
import pandas as pd
from itertools import compress


class ArgoDataset(Dataset):
    def __init__(self, root_dir, root_dir_lane, random_rotation=False, random_translation=False, get_id=False):
        super(ArgoDataset, self).__init__()
        self.loader = DataLaneLoader(root_dir, root_dir_lane)
        self.down_sampling = 1
        self.time_len = 50
        self.hist_len = 20
        self.dist_max = 30
        self.min_num_obs = 4
        self.max_num_lanes = 5
        self.max_size_lane = 70
        self.get_id = get_id
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.translation_distance_std = 10
        # self.city_lanes = {}

    def __len__(self):
        return len(self.loader)

    def _get_lane_by_track_id(self, df, track_id):
        return df[df["TRACK_ID"] == track_id]

    def __getitem__(self, idx):
        track_df = self.loader[idx].seq_df
        lane_df = self.loader[idx].lane_df

        time_list = sorted(track_df["TIMESTAMP"].unique())
        len_time = len(time_list)

        frames = track_df.groupby("TRACK_ID")
        lane_frame = lane_df.groupby("TRACK_ID")

        coor_list = []
        lane_list = []
        mask_lane_list = []
        mask_list = []
        id_list = []
        mask_temp = np.ones([len_time])
        coor_temp = np.zeros([len_time, 2])

        for track_id, track_data in frames:
            mask_temp.fill(1)
            coor_temp.fill(0)
            object_type = track_data["OBJECT_TYPE"].values[0]

            # if object_type == 'OTHER':
            #     continue

            coor_x = track_data["X"].values
            coor_y = track_data["Y"].values

            # lanes_track = self._get_lane_by_track_id(lane_df, track_id)
            try:
                track_lane = lane_frame.get_group(track_id)
                lanes_by_id = track_lane.groupby("LANE_ID")
            except KeyError:
                lanes_by_id = []

            coor_lane_temp = np.zeros([self.max_num_lanes, self.max_size_lane, 2])
            mask_lane_temp = np.zeros([self.max_num_lanes, self.max_size_lane])
            for ind, (_, lane) in enumerate(lanes_by_id):
                if ind >= self.max_num_lanes-1:
                    break
                lane_x = lane["LANE_X"].values
                lane_y = lane["LANE_Y"].values
                len_lane = len(lane_x)
                if len_lane < self.max_size_lane:
                    coor_lane_temp[ind, :len_lane, :] = np.stack([lane_x, lane_y]).transpose()
                    mask_lane_temp[ind, :len_lane] = 1
                else:
                    coor_lane_temp[ind, :, :] = np.stack([lane_x, lane_y]).transpose()[-self.max_size_lane:, :]
                    mask_lane_temp[ind, :] = 1
            time = track_data["TIMESTAMP"].values

            id = track_lane["SEQ_ID"].values[0]
            coor = np.stack([coor_x, coor_y]).transpose()
            if coor.shape[0] == len_time:
                coor_temp = coor
            else:
                args = np.searchsorted(time_list, time)
                mask_temp.fill(0)
                mask_temp[args] = 1
                coor_temp[args, :] = coor

            if object_type == "AGENT":
                coor_list.insert(0, coor_temp.copy())
                mask_list.insert(0, mask_temp.copy())
                lane_list.insert(0, coor_lane_temp.copy())
                mask_lane_list.insert(0, mask_lane_temp.copy())
                id_list.insert(0, id)
            elif object_type == "AV":
                pos_insert = min(len(coor_list), 1)
                coor_list.insert(pos_insert, coor_temp.copy())
                mask_list.insert(pos_insert, mask_temp.copy())
                lane_list.insert(pos_insert, coor_lane_temp.copy())
                mask_lane_list.insert(pos_insert, mask_lane_temp.copy())
                id_list.insert(pos_insert, id)
            else:
                coor_list.append(coor_temp.copy())
                mask_list.append(mask_temp.copy())
                lane_list.append(coor_lane_temp.copy())
                mask_lane_list.append(mask_lane_temp.copy())
                id_list.append(id)

        coor_all = np.stack(coor_list).transpose([1, 0, 2])
        lane_all = np.stack(lane_list).transpose([2, 0, 1, 3])
        mask_lane_all = np.stack(mask_lane_list).transpose([2, 0, 1])
        mask_all = np.stack(mask_list).transpose([1, 0])
        id_all = id_list

        # mean_pos = np.sum(coor_all[19, :, :], axis=0) / np.sum(mask_all[19, :])
        mean_pos = coor_all[19, 0, :]
        # map_x = mean_pos[0]
        # map_y = mean_pos[1]

        rel_coor_all = (coor_all - mean_pos)*(mask_all[:, :, None])
        rel_lane_all = (lane_all - mean_pos)*(mask_lane_all[:, :, :, None])
        # mask vehicle allways further than dist_max or not seen (keep only vehicle seen at less than dist_max)
        rel_dist = np.square(rel_coor_all[:, :, 0]) + np.square(rel_coor_all[:, :, 1])
        mask_dist = np.any(np.logical_and(rel_dist != 0, rel_dist < self.dist_max * self.dist_max), axis=0)
        # mask vehicles that where not observed enough
        mask_obs = np.all(mask_all[17:20, :] != 0, axis=0)
        mask_veh = np.logical_and(mask_dist, mask_obs)
        assert mask_veh[0]
        rel_coor_all = rel_coor_all[:, mask_veh, :]
        id_all = [id for id in compress(id_all, mask_veh)]
        rel_lane_all = rel_lane_all[:, mask_veh, :, :]
        mask_lane_all = mask_lane_all[:, mask_veh, :]
        mask_all = mask_all[:, mask_veh]

        if self.random_translation:
            distance = np.random.normal([0, 0], self.translation_distance_std, 2)
            rel_coor_all = rel_coor_all + mask_all[:, :, None]*distance
            rel_lane_all = rel_lane_all + mask_lane_all[:, :, :, None]*distance

        if self.random_rotation:
            angle = np.random.uniform(0, 2*np.pi)
            rel_coor_all = self.scene_rotation(rel_coor_all, angle)
            rel_lane_all = self.scene_rotation(rel_lane_all, angle)

        if self.get_id :
            return rel_coor_all, mask_all, rel_lane_all, mask_lane_all, id_all
        else:
            return rel_coor_all, mask_all, rel_lane_all, mask_lane_all

    @staticmethod
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

    @staticmethod
    def _count_last_obs(coor, hist_len=None):
        if hist_len is not None:
            time_len_coor = np.sum(np.sum(np.cumprod((coor[hist_len-1::-1] != 0), 0), 2) > 0, 0)
        else:
            time_len_coor = np.sum(np.sum(np.cumprod((coor[coor.shape[0]-1::-1] != 0), 0), 2) > 0, 0)
        return time_len_coor

    def collate_fn(self, samples):
        time_len = self.time_len // self.down_sampling
        hist_len = self.hist_len // self.down_sampling

        max_n_vehicle = 0
        max_n_lanes = 0
        for coor, mask, lanes, mask_lanes in samples:
            # time_len_coor = self._count_last_obs(coor, hist_len*self.down_sampling)
            # num_vehicle = np.sum(time_len_coor > self.min_num_obs)
            num_vehicle = coor.shape[1]
            num_lanes = len(lanes)
            if num_lanes > 0:
                points_len = lanes.shape[1]
            max_n_vehicle = max(num_vehicle, max_n_vehicle)
            max_n_lanes = max(num_lanes, max_n_lanes)
        if max_n_vehicle <= 0:
            raise RuntimeError

        data_batch = np.zeros([time_len, len(samples), max_n_vehicle, 2])
        mask_batch = np.zeros([time_len, len(samples), max_n_vehicle])
        lane_batch = np.zeros([self.max_size_lane, len(samples), max_n_vehicle, self.max_num_lanes, 2])
        mask_lane_batch = np.zeros([self.max_size_lane, len(samples), max_n_vehicle, self.max_num_lanes])

        for sample_ind, (coor, mask, lanes, mask_lanes) in enumerate(samples):
            # args = np.argwhere(self._count_last_obs(coor, hist_len*self.down_sampling) > self.min_num_obs)[:, 0]
            data_batch[:, sample_ind, :coor.shape[1], :] = coor[::self.down_sampling, :, :]
            mask_batch[:, sample_ind, :mask.shape[1]] = mask[::self.down_sampling, :]
            lane_batch[:, sample_ind, :lanes.shape[1], :, :] = lanes
            mask_lane_batch[:, sample_ind, :lanes.shape[1], :] = mask_lanes

        mask_past = mask_batch[:hist_len]

        mask_fut = np.cumprod(mask_batch[hist_len-self.min_num_obs:], 0)[self.min_num_obs:]
        # mask_past = np.cumprod(mask_past[::-1], 0)[::-1] #should not change anything

        data_batch = torch.from_numpy(data_batch.astype('float32'))
        mask_past = torch.from_numpy(mask_past.astype('bool'))
        mask_fut = torch.from_numpy(mask_fut.astype('bool'))
        lane_batch = torch.from_numpy(lane_batch.astype('float32'))
        mask_lane_batch = torch.from_numpy(mask_lane_batch.astype('bool'))

        return data_batch[:hist_len], data_batch[hist_len:], mask_past, mask_fut, lane_batch, mask_lane_batch


