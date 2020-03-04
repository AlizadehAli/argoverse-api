import streamlit as st
from bokeh.plotting import figure
from bokeh.models.glyphs import Rect, Line
from bokeh.models import ColumnDataSource, Arrow, NormalHead
from imitator_simulator import ImitatorSimulator
from torch.utils.data import DataLoader
from get_model import get_model
from pydoc import locate

import numpy as np
import torch

import time

dataset_path = "val/dataset2/"
# dataset_path = "forecasting_sample/dataset2/"

load_model_name = 'sumAttention_openLoop_sepEgo_grudec_prodLane_big_190_bis2'
log_dir = "runs"

batch_size = 1
batch_ind = 0
veh_width = 5
veh_height = 2
len_hist = 20
len_lane = 40
len_pred = 3

net = get_model('runs', load_model_name, load_model_name)
simulator = ImitatorSimulator(net)


def get_dataset(scale_factor, log_dir, load_model_name):
    ArgoDataset = locate(log_dir + '.' + load_model_name + ".dataset_loader.ArgoDataset")
    # ArgoDataset = locate("dataset_loader.ArgoDataset")
    return ArgoDataset(dataset_path, normalize=False,
                        scale_factor=scale_factor, limit_file_number=2)


dataset = get_dataset(1, log_dir, load_model_name)
data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=dataset.collate_fn))


def get_numpy_data(traj_past, traj_fut, pred_fut, mask_past, mask_fut, lanes, mask_lanes):

    traj_past = traj_past.cpu().detach().numpy()
    traj_fut = traj_fut.detach().numpy()
    lanes = lanes.cpu().detach().numpy()
    mask_past = mask_past.cpu().detach().numpy().astype('bool')
    mask_fut = mask_fut.cpu().detach().numpy().astype('bool')
    mask_lanes = mask_lanes.cpu().detach().numpy().astype('bool')
    pred_fut = pred_fut.cpu().detach().numpy()
    pred_fut[:, :, :, :, 5].sort(axis=3)
    pred_fut[:, :, :, :, 5] = np.log(pred_fut[:, :, :, :, 5] * 1e3 + 1)
    pred_fut[:, :, :, :, 5] = pred_fut[:, :, :, :, 5] / np.sum(pred_fut[:, :, :, :, 5], axis=3, keepdims=True)
    vx = traj_past[-1, :, :, 0] - traj_past[-2, :, :, 0]
    vy = traj_past[-1, :, :, 1] - traj_past[-2, :, :, 1]
    v = vx * vx + vy * vy
    mask_angles = v<0.1
    angles = np.arctan2(vy, vx)
    # return traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles, pred_fut[:, :, 0, :, :]
    return (traj_past, traj_fut, mask_past, mask_fut, lanes,
           mask_lanes, angles, mask_angles, pred_fut[:, :, :, :])


def get_data_init():
    traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes = next(data_loader)
    traj_fut = traj_fut[:len_pred]
    mask_fut = mask_fut[:len_pred]

    if torch.cuda.is_available():
        traj_past = traj_past.cuda()
        mask_past = mask_past.cuda()
        lanes = lanes.cuda()
        mask_lanes = mask_lanes.cuda()

    next_pos = simulator.init(traj_past, mask_past, lanes, mask_lanes)
    pred_fut, _ = simulator.get_current_prediction()

    return get_numpy_data(traj_past, traj_fut, pred_fut, mask_past, mask_fut, lanes, mask_lanes)


def step_simulation(prev_pos):
    next_pos = simulator().cpu().detach().numpy()
    pred_fut, mask_fut = simulator.get_current_prediction()
    pred_fut = pred_fut.cpu().detach().numpy()
    pred_fut[:, :, :, :, 5].sort(axis=3)
    pred_fut[:, :, :, :, 5] = np.log(pred_fut[:, :, :, :, 5] * 1e3 + 1)
    pred_fut[:, :, :, :, 5] = pred_fut[:, :, :, :, 5] / np.sum(pred_fut[:, :, :, :, 5], axis=3, keepdims=True)
    vx = next_pos[:, :, :, 0] - prev_pos[:, :, :, 0]
    vy = next_pos[:, :, :, 1] - prev_pos[:, :, :, 1]
    mask_angles = (vx*vx + vy*vy) < 0.1
    angles = np.arctan2(vy, vx)
    return next_pos, pred_fut, mask_fut, angles, mask_angles


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
agent_lane_glyph = Line(x='x', y='y', line_color='red', line_dash='dotted', line_alpha=0.7, line_width=4)
vehicle_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='grey', fill_alpha=0.5, line_color=None)
ego_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='green', fill_alpha=0.5, line_color=None)
agent_glyph = Rect(x='x', y='y', angle='angle', width=veh_width, height=veh_height,
                     fill_color='red', fill_alpha=0.5, line_color=None)
# target_glyph = Circle(x='x', y='y', radius=2, line_color=None, fill_color='green', fill_alpha=0.2)

traj_past, traj_fut, mask_past, mask_fut, lanes, mask_lanes, angles, mask_angles, pred_fut = get_data_init()
# p.add_glyph(pos_circle_source, target_glyph)

chart = None
pred_traj_source = []
vehicle_source = []
for sim_step in range(50):
    print("\r"+str(sim_step)+'/'+str(100), end="")
    # Plot predicted trajectories with blue arrows
    # for i in range(6):
    #     data_dict = []
    #     for j in range(pred_fut.shape[-2]):
    #         data_dict.append(dict(x_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 0][:-1, j],
    #                      y_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 1][:-1, j],
    #                      x_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 0][1:, j],
    #                      y_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 1][1:, j]))
    #     if sim_step == 0:
    #         for j in range(pred_fut.shape[-2]):
    #             pred_traj_source.append(ColumnDataSource(
    #                 data_dict[j]))
    #             alpha = 0.5#np.mean(pred_fut[mask_fut[:, batch_ind, 0], batch_ind, i, 5])
    #             p.add_layout(Arrow(end=NormalHead(fill_alpha=alpha, fill_color='blue',
    #                                               line_color='blue', line_alpha=alpha, size=5),
    #                                x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
    #                                line_color='blue', line_alpha=alpha, line_width=3, source=pred_traj_source[-1]))
    #     else:
    #         for j in range(pred_fut.shape[-2]):
    #             pred_traj_source[i+6*j].stream(data_dict[j], pred_fut.shape[0])


    # #Plot predicted last positions with black arrowsr
    # pred_traj_source = ColumnDataSource(
    #         dict(x_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 0][-2],
    #              y_start=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 1][-2],
    #              x_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 0][-1],
    #              y_end=pred_fut[mask_fut[:, batch_ind, 0], batch_ind, :, 1][-1]))
    # p.add_layout(Arrow(end=NormalHead(fill_alpha=1, fill_color='black',
    #                                   line_color='black', line_alpha=1, size=7),
    #                    x_start='x_start', y_start='y_start', x_end='x_end', y_end='y_end',
    #                    line_color='blue', line_alpha=1, line_width=3, source=pred_traj_source))

    #Plot vehicles
    for veh_ind in range(traj_past.shape[2]):
        if mask_fut[0, batch_ind, veh_ind]:
            if sim_step == 0:
                data_dict = dict(x=traj_past[-1, batch_ind, veh_ind:veh_ind + 1, 0],
                                 y=traj_past[-1, batch_ind, veh_ind:veh_ind + 1, 1],
                                 angle=angles[batch_ind, veh_ind:veh_ind + 1])
                vehicle_source.append(ColumnDataSource(data_dict))
                if veh_ind == 0:
                    p.add_glyph(vehicle_source[-1], agent_glyph)
                elif veh_ind == 1:
                    p.add_glyph(vehicle_source[-1], ego_glyph)
                else:
                    p.add_glyph(vehicle_source[-1], vehicle_glyph)
            else:
                data_dict = dict(x=next_pos[-1, batch_ind, veh_ind:veh_ind + 1, 0],
                                 y=next_pos[-1, batch_ind, veh_ind:veh_ind + 1, 1],
                                 angle=angles[-1, batch_ind, veh_ind:veh_ind + 1])
                vehicle_source[veh_ind].stream(data_dict, 1)
        else:
            if sim_step == 0:
                data_dict = dict(x=traj_past[-1, batch_ind, veh_ind:veh_ind + 1, 0],
                                 y=traj_past[-1, batch_ind, veh_ind:veh_ind + 1, 1],
                                 angle=angles[batch_ind, veh_ind:veh_ind + 1])
                vehicle_source.append(ColumnDataSource(data_dict))
            else:
                vehicle_source[veh_ind].data['x'] = [np.nan]
                vehicle_source[veh_ind].data['y'] = [np.nan]
                vehicle_source[veh_ind].data['angle'] = [np.nan]


    #Plot lanes
    for lane_ind in range(lanes.shape[2]):
        lane_source = ColumnDataSource(
            dict(x=lanes[mask_lanes[:, batch_ind, lane_ind], batch_ind, lane_ind, 0],
                 y=lanes[mask_lanes[:, batch_ind, lane_ind], batch_ind, lane_ind, 1]))
        if sim_step == 0:
            p.add_glyph(lane_source, lane_glyph)

    if sim_step == 0:
        p.add_glyph(lane_source, agent_lane_glyph)
        chart = st.bokeh_chart(p)
        max_n_veh = simulator.max_num_veh
        n_veh = min(traj_past.shape[2], max_n_veh)
        padding = max_n_veh - n_veh

        next_pos, pred_fut, mask_fut, angles, mask_angles = step_simulation(
            np.pad(traj_past[-1:], ((0, 0), (0, 0), (0, padding), (0, 0)), mode='constant', constant_values=0))
    else:
        next_pos, pred_fut, mask_fut, angles, mask_angles = step_simulation(next_pos)
        chart.bokeh_chart(p)
        # chart = st.bokeh_chart(p)
    # time.sleep(0.1)



