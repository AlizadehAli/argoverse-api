import streamlit as st
from bokeh.plotting import figure
from bokeh.models.glyphs import Rect, Line, Circle
from bokeh.models import ColumnDataSource, Arrow, NormalHead

import numpy as np
import pandas
import torch
import os, sys

from torch.utils.data import DataLoader

from get_model import get_model

from gmm_to_path.find_modes import get_gmm_modes
from fusion_dataset_loader import FusionDataset

st.title('Plot sample trajectories from argoverse')

# dataset_path = "forecasting_sample/dataset2/"
dataset_path = "val/dataset2/"

# @st.cache
def get_dataset2(scale_factor, log_dir, load_model_name):
    from pydoc import locate
    # ArgoDataset = locate(log_dir + '.' + load_model_name + ".dataset_loader.ArgoDataset")
    ArgoDataset = locate("dataset_loader.ArgoDataset")
    return ArgoDataset(dataset_path, normalize=True,
                       scale_factor=scale_factor, limit_file_number=2)

def get_dataset(scale_factor, log_dir, load_model_name):
    # current_path = os.getcwd()
    # sys.path.insert(1, current_path + '/../')
    return FusionDataset('../data_fusion/SUV_TF_60kkm/test_sequenced_data.tar')


# load_model_name = 'model_allGeomAttention_103'
log_dir = "runs"
dir_list = [dI for dI in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir,dI))]
dir_list.sort(key=lambda dI: os.stat(os.path.join(log_dir, dI)).st_mtime, reverse=True)
load_model_name = st.selectbox("Select model:", dir_list)
open_loop = 'openloop' in load_model_name.lower()
cylindric = 'cylindric' in load_model_name.lower()

def get_net(load_model_name):
    return get_model('runs', load_model_name, load_model_name, False, len_hist, len_lane, len_pred)

batch_size = 1
veh_width = 4
veh_height = 2
len_hist = 25
len_lane = 40
len_pred = 37

net = get_net(load_model_name)
# net.set_training(True)
if hasattr(net, 'scale_factor'):
    scale_factor = net.scale_factor
else:
    scale_factor = 1
dataset = get_dataset(scale_factor, log_dir, load_model_name)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=dataset.collate_fn)

def get_index(index=None):
    if index is None:
        index = np.random.randint(0, len(data_loader)-1)
    elif isinstance(index, str):
        try:
            index = int(index)
        except:
            index = None
    if not isinstance(index, int) or not (0 < index < len(data_loader) -1):
        index = np.random.randint(0, len(data_loader) - 1)
    return index

def get_data(index):
    n_points_slope = 19
    st.write("Sequence ID: %d" % index)
    try:
        traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, indices = dataset.collate_fn([dataset[index]])
    except:
        traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes = dataset.collate_fn([dataset[index]])

    if traj_past.shape[2] == 1:
        traj_past = torch.cat((traj_past, traj_past), dim=2)
        mask_past = torch.cat((mask_past, torch.zeros_like(mask_past)), dim=2)

    if torch.cuda.is_available():
        traj_past = traj_past.cuda()
        mask_past = mask_past.cuda()
        lanes = lanes.cuda()
        mask_lanes = mask_lanes.cuda()

    if open_loop:
        current_pos = traj_past[-1:]
        traj_past_diff = traj_past[1:] - traj_past[:-1]
        try:
            pred_fut, _ = net(traj_past_diff, mask_past,
                           lanes, mask_lanes,
                           len_pred, init_pos=current_pos)
        except:
            pred_fut = net(traj_past_diff, mask_past,
                              lanes, mask_lanes,
                              len_pred, init_pos=current_pos)
        if cylindric:
            x = pred_fut[:, :, :, :, 0:1]*torch.cos(pred_fut[:, :, :, :, 1:2])
            y = pred_fut[:, :, :, :, 0:1]*torch.sin(pred_fut[:, :, :, :, 1:2])
            pred_fut[:, :, :, :, :2] = torch.cat((x, y), dim=-1)
        pred_fut[:, :, :, :, :2] = \
            torch.cumsum(pred_fut[:, :, :, :, :2], dim=0) + current_pos[:, :, :1, :].unsqueeze(3)

    else:
        pred_fut = net(traj_past, mask_past,
                       lanes, mask_lanes, len_pred)


    traj_past = traj_past.cpu().detach().numpy()
    traj_fut = traj_fut.detach().numpy()
    lanes = lanes.cpu().detach().numpy()
    mask_past = mask_past.cpu().detach().numpy().astype('bool')
    mask_fut = mask_fut.detach().numpy().astype('bool')
    mask_lanes = mask_lanes.cpu().detach().numpy().astype('bool')
    pred_fut = pred_fut.cpu().detach().numpy()
    # pred_fut[:, :, :, :, 5].sort(axis=3)
    main_mode = get_gmm_modes(pred_fut)
    pred_fut[:, :, :, :, 5] = np.log(pred_fut[:, :, :, :, 5]*1e2 + 1)
    pred_fut[:, :, :, :, 5] = pred_fut[:, :, :, :, 5]/np.sum(pred_fut[:, :, :, :, 5], axis=3, keepdims=True)
    vx = np.mean(traj_past[-n_points_slope:, :, :, 0] - traj_past[-(n_points_slope+1):-1, :, :, 0], axis=0)
    vy = np.mean(traj_past[-n_points_slope:, :, :, 1] - traj_past[-(n_points_slope+1):-1, :, :, 1], axis=0)
    angles = np.arctan2(vy, vx)
    # return traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles, pred_fut[:, :, 0, :, :]
    return traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles, pred_fut[:, :, 0, :], main_mode[:, :, 0, :]

# @st.cache
# def get_sample():

# dataset = ArgoDataset(data_dir, lane_data_dir, random_rotation=False, random_translation=False)
# data_ind = np.random.randint(0, len(dataset)-1)
# # traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles = get_data()
# traj, mask_traj, lanes, mask_lanes = dataset[data_ind]
# mask_traj = mask_traj.astype('bool')
# mask_lanes = mask_lanes.astype('bool')
# traj_past = np.expand_dims(traj[:20], axis=1)
# traj_fut = np.expand_dims(traj[20:], axis=1)
# mask_past = np.expand_dims(mask_traj[:20], axis=1)
# mask_fut = np.expand_dims(mask_traj[20:], axis=1)
# lanes = np.expand_dims(lanes, axis=1)
# mask_lanes = np.expand_dims(mask_lanes, axis=1)
# vx = traj_past[-1, :, :, 0] - traj_past[-2, :, :, 0]
# vy = traj_past[-1, :, :, 1] - traj_past[-2, :, :, 1]
# angles = np.arctan2(vy, vx)
index_box = st.text_input("Sequence ID to plot:", "Random")
index_int = get_index(index_box)
# index_int = 2679
traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles, pred_fut, modes = get_data(index_int)

batch_ind = 0#np.random.randint(0, traj_past.shape[1]-1)
p = figure(title="Dataset sample",
           x_axis_label='x',
           y_axis_label='y',
           match_aspect=True,
           sizing_mode='stretch_both',
           active_scroll='wheel_zoom')

# past_traj_glyph = Arrow(end=NormalHead(), x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
#                         line_color='grey', line_width=3)
# past_traj_glyph = Line(x='x', y='y', line_color='grey', line_width=3)
fut_traj_glyph = Line(x='x', y='y', line_color='green', line_width=3)
lane_glyph = Line(x='x', y='y', line_color='black', line_dash='dashed')
vehicle_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='grey', fill_alpha=0.5, line_color=None)
ego_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='green', fill_alpha=0.5, line_color=None)
agent_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='red', fill_alpha=0.5, line_color=None)
# target_glyph = Circle(x='x', y='y', radius=2, line_color=None, fill_color='green', fill_alpha=0.2)

p.circle(traj_fut[mask_fut[:, batch_ind, 0], batch_ind, 0, 0][-1:],
         traj_fut[mask_fut[:, batch_ind, 0], batch_ind, 0, 1][-1:],
         radius=2, line_color=None, fill_color='green', fill_alpha=0.2)


# p.add_glyph(pos_circle_source, target_glyph)

for i in range(6):
    pred_traj_source = ColumnDataSource(
        dict(x_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 0][:-1],
             y_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 1][:-1],
             x_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 0][1:],
             y_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 1][1:]))
    alpha = np.mean(pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 5])
    p.add_layout(Arrow(end=NormalHead(fill_alpha=alpha, fill_color='blue',
                                      line_color='blue', line_alpha=alpha, size=5),
                       x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
                       line_color='blue', line_alpha=alpha, line_width=3, source=pred_traj_source))
# print modes
# for i in range(modes.shape[-2]):
#     pred_traj_source = ColumnDataSource(
#         dict(x_start=modes[mask_fut[:, batch_ind, 0], batch_ind, i, 0][:-1],
#              y_start=modes[mask_fut[:, batch_ind, 0], batch_ind, i, 1][:-1],
#              x_end=modes[mask_fut[:, batch_ind, 0], batch_ind, i, 0][1:],
#              y_end=modes[mask_fut[:, batch_ind, 0], batch_ind, i, 1][1:]))
#     p.add_layout(Arrow(end=NormalHead(fill_alpha=0.8, fill_color='black',
#                                       line_color='black', line_alpha=0.8, size=5),
#                        x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
#                        line_color='black', line_alpha=0.8, line_width=3, source=pred_traj_source))
pred_traj_source = ColumnDataSource(
        dict(x_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 0][-2],
             y_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 1][-2],
             x_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 0][-1],
             y_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 1][-1]))
p.add_layout(Arrow(end=NormalHead(fill_alpha=1, fill_color='black',
                                  line_color='black', line_alpha=1, size=7),
                   x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
                   line_color='blue', line_alpha=1, line_width=3, source=pred_traj_source))

    # p.add_glyph(pred_traj_source, pred_traj_glyph)
    # p.line(pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 0],
    #        pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 1],
    #        line_alpha=np.mean(pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 5]),
    #        line_color='blue', line_width=5)
# p.line(pred_fut[mask_fut[:, batch_ind, 0], batch_ind, 0],
#        pred_fut[mask_fut[:, batch_ind, 0], batch_ind, 1],
#        line_color='blue', line_width=5)
fut_traj_source = ColumnDataSource(
    dict(x=traj_fut[mask_fut[:, batch_ind, 0], batch_ind, 0, 0],
         y=traj_fut[mask_fut[:, batch_ind, 0], batch_ind, 0, 1]))
p.add_glyph(fut_traj_source, fut_traj_glyph)

for veh_ind in range(traj_fut.shape[2]):
    past_traj_source = ColumnDataSource(
        dict(x_start=traj_past[mask_past[:, batch_ind, veh_ind], batch_ind, veh_ind, 0][:-1],
             y_start=traj_past[mask_past[:, batch_ind, veh_ind], batch_ind, veh_ind, 1][:-1],
             x_end=traj_past[mask_past[:, batch_ind, veh_ind], batch_ind, veh_ind, 0][1:],
             y_end=traj_past[mask_past[:, batch_ind, veh_ind], batch_ind, veh_ind, 1][1:]))
    p.add_layout(Arrow(end=NormalHead(fill_alpha=0.5, fill_color='grey',
                                      line_color='grey', line_alpha=0.5, size=5),
                       x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
                       line_color='grey', line_alpha=0.5, line_width=3, source=past_traj_source))

    if veh_ind > 0:
        fut_traj_source = ColumnDataSource(
            dict(x_start=traj_fut[mask_fut[:, batch_ind, veh_ind], batch_ind, veh_ind, 0][:-1],
                 y_start=traj_fut[mask_fut[:, batch_ind, veh_ind], batch_ind, veh_ind, 1][:-1],
                 x_end=traj_fut[mask_fut[:, batch_ind, veh_ind], batch_ind, veh_ind, 0][1:],
                 y_end=traj_fut[mask_fut[:, batch_ind, veh_ind], batch_ind, veh_ind, 1][1:]))
        p.add_layout(Arrow(end=NormalHead(fill_alpha=0.5, fill_color='grey',
                                          line_color='grey', line_alpha=0.5, size=5),
                           x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
                           line_color='grey', line_width=3, line_alpha=0.5, source=fut_traj_source))

    if mask_fut[0, batch_ind, veh_ind]:
        vehicle_source = ColumnDataSource(
            dict(x=traj_past[-1, batch_ind, veh_ind:veh_ind+1, 0],
                 y=traj_past[-1, batch_ind, veh_ind:veh_ind+1, 1],
                 angle=angles[batch_ind, veh_ind:veh_ind+1]))
        if veh_ind == 0:
            p.add_glyph(vehicle_source, agent_glyph)
        elif veh_ind == 1:
            p.add_glyph(vehicle_source, ego_glyph)
        else:
            p.add_glyph(vehicle_source, vehicle_glyph)

for lane_ind in range(lanes.shape[2]):
    lane_source = ColumnDataSource(
        dict(x=lanes[mask_lanes[:, batch_ind, lane_ind], batch_ind, lane_ind, 0],
             y=lanes[mask_lanes[:, batch_ind, lane_ind], batch_ind, lane_ind, 1]))
    p.add_glyph(lane_source, lane_glyph)


st.bokeh_chart(p)

