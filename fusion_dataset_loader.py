from fusion_indices import *
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class FusionDataset(Dataset):
    def __init__(self, dataset_path, random_rotation=False, random_translation=False):
        super(FusionDataset, self).__init__()
        self.load_dataset(dataset_path)
        self.random_rotation = random_rotation
        self.random_translation = random_translation
        self.translation_distance_std = 2
        self.data_to_get = [OBJ_X, OBJ_Y, OBJ_CLASS]
        self.line_data_to_get = [0, 1]
        self.hist_len = int(1000*2/40)
        self.fut_len = int(1000*3/40)
        self.time_len = self.hist_len + self.fut_len
        self.min_num_obs = 10
        self.down_sampling = 1

    def load_dataset(self, path):
        with open(path, 'rb') as handle:
            self.dataset = pickle.load(handle)

    def __len__(self):
        return len(self.dataset['traj'])

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

    def __getitem__(self, item):
        traj = self.dataset['traj'][item][:, :, self.data_to_get]
        lines = self.dataset['lines'][item][:, :, self.line_data_to_get]

        if self.random_translation:
            distance = np.random.normal([0, 0], self.translation_distance_std, 2)
            traj[:, :, :2] = traj[:, :, :2] + distance
            lines[:, :, :2] = lines[:, :, :2] + distance

        if self.random_rotation:
            angle = np.random.uniform(0, 2 * np.pi)
            traj[:, :, :2] = self.scene_rotation(traj[:, :, :2], angle)
            # lines[:, :, :2] = self.scene_rotation(lines[:, :, :2], angle)

        return traj, self.dataset['mask_traj'][item], lines, self.dataset['mask_lines'][item]

    def collate_fn(self, samples):
        max_n_veh = 0
        max_n_lines = 0
        for traj, mask, lines, mask_lines in samples:
            num_veh = traj.shape[1]
            num_lines = lines.shape[1]
            max_n_veh = max(max_n_veh, num_veh)
            max_n_lines = max(max_n_lines, num_lines)
        time_len = traj.shape[0]
        assert time_len == self.time_len
        time_len = self.time_len // self.down_sampling
        hist_len = self.hist_len // self.down_sampling
        data_batch = np.zeros([time_len, len(samples), max_n_veh, len(self.data_to_get)])
        mask_batch = np.zeros([time_len, len(samples), max_n_veh], dtype='bool')
        lines_batch = np.zeros([10, len(samples), max_n_lines, len(self.line_data_to_get)])
        mask_lines_batch = np.zeros([10, len(samples), max_n_lines], dtype='bool')

        for i, (traj, mask, lines, mask_lines) in enumerate(samples):
            data_batch[:, i, :traj.shape[1], :] = traj[:self.down_sampling*time_len:self.down_sampling, :, :]
            mask_batch[:, i, :mask.shape[1]] = mask[:self.down_sampling*time_len:self.down_sampling, :]
            lines_batch[:, i, :lines.shape[1], :] = lines[:, :, :]
            mask_lines_batch[:, i, :mask_lines.shape[1]] = mask_lines[:, :]

        mask_past = mask_batch[:hist_len]
        mask_fut = np.cumprod(mask_batch[hist_len-self.min_num_obs:], 0)[self.min_num_obs:]

        data_batch = torch.from_numpy(data_batch.astype('float32'))
        mask_past = torch.from_numpy(mask_past.astype('bool'))
        mask_fut = torch.from_numpy(mask_fut.astype('bool'))
        lines_batch = torch.from_numpy(lines_batch.astype('float32'))
        mask_lines_batch = torch.from_numpy(mask_lines_batch.astype('bool'))

        return data_batch[:hist_len], data_batch[hist_len:], mask_past, mask_fut, lines_batch, mask_lines_batch
