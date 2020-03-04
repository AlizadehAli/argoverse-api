import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib._color_data as mcd
from fusion_indices import *
from clothoid import Clothoidxy

data_dir = '../data_fusion/SUV_TF_60kkm/'

file_list = [f for f in glob.iglob(data_dir + '**/*.csv', recursive=True)]

hist_time_sec = 2
fut_time_sec = 3

# Read all lines in all files and append it in data
data = []
for file in file_list[:]:
    with open(file) as f:
        lines = f.readlines()
        list_data = [np.fromstring(line, dtype=float, sep=',') for line in lines]
    data.append(list_data)

# Get the time steps and the number of steps in the desired sequences
dt = data[0][101][GLOBAL_TIME] - data[0][100][GLOBAL_TIME]
hist_size = int(np.ceil((1000*hist_time_sec)/dt))
# there could be some redundancies between the end of one file and the begining of the next one
# discare_size = int(np.ceil((1000*20)/dt))
discare_size = 0
size_sequence = int(np.ceil((1000*(hist_time_sec + fut_time_sec))/dt))

## Read the data and produce sequences
train_traj_list = []
train_mask_list = []
train_line_list = []
train_mask_line_list = []
train_path_list = []
val_traj_list = []
val_mask_list = []
val_line_list = []
val_mask_line_list = []
val_path_list = []
test_traj_list = []
test_mask_list = []
test_line_list = []
test_mask_line_list = []
test_path_list = []

val_ratio = 0.1
test_ratio = 0.2
#train set is a third at the begining of the sequence, a third between test and validation and a third in the end
for run, path in zip(data, file_list):
    n_sequence = (len(run) - discare_size) // size_sequence
    size_val = int(val_ratio * n_sequence)
    size_test = int(test_ratio * n_sequence)
    size_train = n_sequence - size_test - size_val

    first_index_test = size_train // 3
    first_index_val = 2 * size_train // 3 + size_test


    for sequence_id in range(n_sequence):
        index = sequence_id*size_sequence

        ### get unique ids of objects and lines
        id_set = set()
        line_id_set = set()
        for t in range(size_sequence):
            n_obj = int(run[index + t][N_OBJ])
            if n_obj > 0:
                id_set.update(run[index + t][OBJ_OFFSET:n_obj*OBJ_DATA_LEN + OBJ_ID + OBJ_OFFSET:OBJ_DATA_LEN])
            line_offset = n_obj * OBJ_DATA_LEN + OBJ_OFFSET + 1
            n_line = int(run[index + t][line_offset - 1])
        n_obj = int(run[index + hist_size - 1][N_OBJ])
        line_offset = n_obj * OBJ_DATA_LEN + OBJ_OFFSET + 1
        n_line = int(run[index + hist_size - 1][line_offset - 1])
        if n_line > 0:
            line_id_set.update(
                run[index + hist_size - 1][line_offset:line_offset + LINE_DATA_LEN * n_line + LINE_ID:LINE_DATA_LEN])
        try:
            id_set.remove(0)
        except KeyError:
            pass
        try:
            line_id_set.remove(0)
        except KeyError:
            pass
        id_list = list(id_set)
        id_list.insert(0, 0.0)
        max_n_obj = len(id_list)
        line_id_list = list(line_id_set)
        max_n_line = len(line_id_list)

        ### numpy arrays to store the data with the size to store each
        ### different observed lines and objects during the sequence
        traj = np.zeros([size_sequence, max_n_obj, OBJ_DATA_LEN])
        line = np.zeros([10, max_n_line, 2])
        mask = np.zeros([size_sequence, max_n_obj], dtype='bool')
        mask_line = np.zeros([10, max_n_line], dtype='bool')

        #### index of the first line data after object data
        n_obj = int(run[index + hist_size-1][N_OBJ])
        line_offset = n_obj * OBJ_DATA_LEN + OBJ_OFFSET + 1
        n_line = int(run[index + hist_size-1][line_offset - 1])
        if n_line > 0:
            line_ids = run[index + hist_size-1][line_offset:line_offset + LINE_DATA_LEN * n_line + LINE_ID:LINE_DATA_LEN]
            args_line_ids = [line_id_list.index(line_id) for line_id in line_ids]
            data_line = run[index + hist_size-1][line_offset:line_offset + LINE_DATA_LEN * n_line].reshape(
                [n_line, LINE_DATA_LEN])
            lines_xy = []
            for nl in range(n_line):
                line_clothoid = Clothoidxy.from_np(
                    data_line[nl, [LINE_X, LINE_Y, LINE_THETA, LINE_C, LINE_DC, LINE_LENGTH]])
                longi = np.arange(10)/10*data_line[nl, LINE_LENGTH]
                line_xy = np.zeros((10, 1, 2))
                for i in range(10):
                    line_xy[i, 0, :] = line_clothoid.p(longi[i])
                lines_xy.append(line_xy)

            line[:, args_line_ids, :] = np.concatenate(lines_xy, axis=1)
            mask_line[:, args_line_ids] = (line_ids > 0)[None, :]


        for t in range(size_sequence):
            n_obj = int(run[index + t][N_OBJ])

            #### Put ego data in the first traj
            traj[t, 0, :EGO_DATA_LEN] = run[index + t][EGO_OFFSET:EGO_DATA_LEN + EGO_OFFSET]
            #### Ego is always there
            mask[t, 0] = 1
            if n_obj > 0:
                ##### All objects ids at time t, (first id is at OBJ_OFFSET+OBJ_ID, then ids are at every OBJ_DATA_LEN)
                obj_ids = run[index + t][OBJ_OFFSET + OBJ_ID:n_obj * OBJ_DATA_LEN + OBJ_ID + OBJ_OFFSET:OBJ_DATA_LEN]
                ##### Get the indices of the objects to store it at the correct place in the numpy array
                args_obj_ids = [id_list.index(obj_id) for obj_id in obj_ids]
                ##### Put object data in the correct place of the traj array
                traj[t, args_obj_ids, :] = run[index + t][OBJ_OFFSET:n_obj * OBJ_DATA_LEN + OBJ_OFFSET].reshape([n_obj, OBJ_DATA_LEN])
                ##### If the object id is greater than 0, the object is there
                mask[t, args_obj_ids] = (obj_ids > 0)

        if first_index_test < sequence_id < first_index_test + size_test:
            test_traj_list.append(traj)
            test_path_list.append(path)
            test_mask_list.append(mask)
            test_line_list.append(line)
            test_mask_line_list.append(mask_line)
        elif first_index_val < sequence_id < first_index_val + size_val:
            val_traj_list.append(traj)
            val_path_list.append(path)
            val_mask_list.append(mask)
            val_line_list.append(line)
            val_mask_line_list.append(mask_line)
        else:
            train_traj_list.append(traj)
            train_path_list.append(path)
            train_mask_list.append(mask)
            train_line_list.append(line)
            train_mask_line_list.append(mask_line)


# Transform to absolute coordinate centered at the ego postion at time 0 (end of history)
def transform(traj_list, mask_list, line_list, mask_line_list):
    for i, (traj, mask, line, mask_line) in enumerate(zip(traj_list, mask_list, line_list, mask_line_list)):
        ## Align EGO and OBJ data
        traj[:, 0, [OBJ_ID, OBJ_X, OBJ_Y, OBJ_YAW, OBJ_VX, OBJ_VY]] = \
            traj[:, 0, [EGO_ID, EGO_DX, EGO_DY, EGO_DYAW, EGO_VX, EGO_VY]].copy()

        ego = traj[:, 0, :].copy()
        ## ego angle is the cumulative sum of its rotations
        angles = np.cumsum(ego[:, OBJ_YAW], axis=0)
        ## setting reference angle at the end of the history sequence
        angles = angles - angles[hist_size-1]

        ## coordinates and velocities rotation at the global sequence orientation
        traj[:, :, [OBJ_X, OBJ_Y]] = sequence_rotation(traj[:, :, [OBJ_X, OBJ_Y]], angles)
        traj[:, :, [OBJ_VX, OBJ_VY]] = sequence_rotation(traj[:, :, [OBJ_VX, OBJ_VY]], angles)

        ego = traj[:, 0, :].copy()
        ## Ego trajectory is the cumsum of its displacements
        ego[:, [OBJ_X, OBJ_Y]] = np.cumsum(ego[:, [OBJ_X, OBJ_Y]], axis=0)
        ## setting the reference position at the end of the history sequence
        ego[:, [OBJ_X, OBJ_Y]] = ego[:, [OBJ_X, OBJ_Y]] - ego[hist_size-1, [OBJ_X, OBJ_Y]]
        ego[:, OBJ_YAW] = angles

        ## Translate the observations to the
        xyyaw = [OBJ_X, OBJ_Y, OBJ_YAW]
        veh = traj[:, 1:, :]
        veh[:, :, xyyaw] = (veh[:, :, xyyaw] + ego[:, None, xyyaw])*mask[:, 1:, None].astype('int')

        traj_list[i][:, 0, :] = ego
        traj_list[i][:, 1:, :] = veh
    return traj_list, mask_list, line_list, mask_line_list

train_traj_list, train_mask_list, train_line_list, train_mask_line_list = transform(train_traj_list, train_mask_list, train_line_list, train_mask_line_list)
test_traj_list, test_mask_list, test_line_list, test_mask_line_list = transform(test_traj_list, test_mask_list, test_line_list, test_mask_line_list)
val_traj_list, val_mask_list, val_line_list, val_mask_line_list = transform(val_traj_list, val_mask_list, val_line_list, val_mask_line_list)

pickle.dump({'traj': train_traj_list, 'mask_traj': train_mask_list, 'lines': train_line_list, 'mask_lines': train_mask_line_list, 'path': train_path_list},
            open(data_dir + "train_sequenced_data.tar", "wb"),  protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump({'traj': test_traj_list, 'mask_traj': test_mask_list, 'lines': test_line_list, 'mask_lines': test_mask_line_list, 'path': test_path_list},
            open(data_dir + "test_sequenced_data.tar", "wb"),  protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump({'traj': val_traj_list, 'mask_traj': val_mask_list, 'lines': val_line_list, 'mask_lines': val_mask_line_list, 'path': val_path_list},
            open(data_dir + "val_sequenced_data.tar", "wb"),  protocol=pickle.HIGHEST_PROTOCOL)
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#
#     for i in range(veh.shape[1]):
#         plt.plot(veh[mask[:, i+1], i, 1], veh[mask[:, i+1], i, 2], '-o',
#                  color=colors[int(veh[hist_size, i, OBJ_CLASS])%len(colors)])
#         # plt.plot(veh[:, i, 1], veh[:, i, 2], '-|', color='gray')
#     for i in range(line.shape[1]):
#         plt.plot(line[mask_line[:, i], i, 1], line[mask_line[:, i], i, 2], '-|', color='black')
#         # plt.plot(line[:, i, 1], line[:, i, 2], '-|', color='black')
#     plt.plot(ego[:, 1], ego[:, 2], '-|', color='r')
#     plt.show()
