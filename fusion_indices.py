import numpy as np

## Indices of the global information in the csv files
GLOBAL_TIME = 0
EGO_OFFSET = 1
N_OBJ = 7

## Indices of the ego information in the csv files
EGO_ID = 0
EGO_DX = 1
EGO_DY = 2
EGO_DYAW = 3
EGO_VX = 4
EGO_VY = 5
EGO_DATA_LEN = 6


## After global and ego information, n_obj times this amount of info about neighbor objects is available
OBJ_OFFSET = 8
OBJ_ID = 0
OBJ_X = 1
OBJ_Y = 2
OBJ_YAW = 3
OBJ_WIDTH = 4
OBJ_LENGTH = 5
OBJ_VX = 6
OBJ_VY = 7
OBJ_CLASS = 8 # 0: car, 1: bus, 2: truck, 3: motorcycle, 4: bicycle, 5: pedestrian, 6: other, 7: unknown small, 8: unknown big, 9: unknown 2 wheels, 10: unknown
OBJ_DATA_LEN = 9

## After object information, the numbner of lines n_line is written then n_lines times this info is available
LINE_ID = 0
LINE_X = 1
LINE_Y = 2
LINE_THETA = 3
LINE_C = 4
LINE_DC = 5
LINE_LENGTH = 6
LINE_DATA_LEN = 7

def sequence_rotation(coor, angle):
    rot_matrix = np.zeros((angle.shape[0], 2, 2))
    c = np.cos(angle)
    s = np.sin(angle)
    rot_matrix[:, 0, 0] = c
    rot_matrix[:, 0, 1] = -s
    rot_matrix[:, 1, 0] = s
    rot_matrix[:, 1, 1] = c
    coor = np.matmul(rot_matrix[:, None, :, :], coor[:, :, :, None]).squeeze(-1)
    return coor
