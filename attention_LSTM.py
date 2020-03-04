import torch
import torch.nn as nn
import torch.functional as F
import torch.jit as jit
import math

from attention_predictor import *

# from attention import attention, LaneAttention, SocialAttention
from activation import Mish, GELU

import os
from pydoc import locate
current_file = os.path.dirname(__file__)
current_path = os.getcwd()

current_file = current_file.replace(current_path, '').replace("/", ".")[1:]
LaneRelationalAttention = locate(current_file + ".attention.LaneRelationalAttention")
SocialRelationalAttention = locate(current_file + ".attention.SocialRelationalAttention")
geometric_relational_attention = locate(current_file + ".attention.geometric_relational_attention")
geometric_attention = locate(current_file + ".attention.geometric_attention")
AttentionBloc = locate(current_file + ".attention.AttentionBloc")

class AttentionLSTM(nn.Module):
    def __init__(self, feature_size, n_heads, lane_feature_size):
        super(AttentionLSTM, self).__init__()
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size
        self.layer_norm_ego = nn.LayerNorm(self.feature_size)
        self.layer_norm = nn.LayerNorm(self.feature_size)
        self.cell1 = nn.LSTMCell(self.feature_size, self.feature_size, True)
        # self.cell2 = nn.LSTMCell(self.feature_size, self.feature_size, True)
        self.cell_ego1 = nn.LSTMCell(self.feature_size, self.feature_size, True)
        # self.cell_ego2 = nn.LSTMCell(self.feature_size, self.feature_size, True)
        # self.attention = jit.script(AttentionBloc(self.feature_size, self.lane_feature_size, n_heads))
        self.attention = AttentionBloc(self.feature_size, self.lane_feature_size, n_heads)

    def forward(self, h_ego, h_pos_ego, state_ego, h, h_pos_veh, state, mask_input, embedded_lanes, lane_mask, batch_size, n_vehicles):
    # type: (Tensor, Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        hx_ego, cx_ego = state_ego
        hx, cx = state

        h_ego, h = self.attention(h_ego, h_pos_ego, h, h_pos_veh, mask_input, embedded_lanes, lane_mask)
        h_ego = self.layer_norm_ego(h_ego.view(batch_size, self.feature_size))
        h = self.layer_norm(h.view(batch_size*n_vehicles, self.feature_size))
        hx_ego, cx_ego = self.cell_ego1(h_ego, (hx_ego, cx_ego))
        # hx_ego, cx_ego = self.cell_ego2(hx_ego, (hx_ego, cx_ego))
        hx, cx = self.cell1(h, (hx, cx))
        # hx, cx = self.cell2(hx, (hx, cx))
        return hx_ego, cx_ego, hx, cx


class AttentionPredictorLSTM(nn.Module, ParametersAttentionPredictor):

    def __init__(self):
        nn.Module.__init__(self)
        ParametersAttentionPredictor.__init__(self)
        self.kernel_size = 3
        self.conv_input = nn.Conv2d(2, self.feature_size, (1, self.kernel_size), padding=(0, 0), stride=(1, 1))
        self.conv_input_ego = nn.Conv2d(2, self.feature_size, (1, self.kernel_size), padding=(0, 0), stride=(1, 1))

        self.lane_embedding = LaneEmbedding(self.lane_feature_size, self.n_lane_layers)

        # self.attention_lstm = jit.script(AttentionLSTM(self.feature_size, self.n_heads_past, self.lane_feature_size))
        self.attention_lstm = AttentionLSTM(self.feature_size, self.n_heads_past, self.lane_feature_size)

        self.output_layer_ego = jit.script(MultiOutputLayer(self.feature_size, self.num_preds))
        self.output_layer = jit.script(MultiOutputLayer(self.feature_size, self.num_preds))

    def _init_state(self, batch_size, n_vehicles):
        self.hx = torch.zeros(batch_size * n_vehicles, self.feature_size)
        self.cx = torch.zeros(batch_size * n_vehicles, self.feature_size)
        self.hx_ego = torch.zeros(batch_size, self.feature_size)
        self.cx_ego = torch.zeros(batch_size, self.feature_size)
        if torch.cuda.is_available():
            self.hx_ego = self.hx_ego.cuda()
            self.cx_ego = self.cx_ego.cuda()
            self.hx = self.hx.cuda()
            self.cx = self.cx.cuda()

    def _to_xy(self, tensor, yaw, dim):
        dl = tensor.narrow(dim, 0, 1)
        x = dl.clone()*torch.cos(yaw.clone())
        y = dl.clone()*torch.sin(yaw.clone())

        return torch.cat((x, y), dim=-1)
        # return tensor.narrow(dim, 0, 2)

    # @torch.jit.export
    def forward(self, input, mask_input, lane_input, lane_mask, len_pred, init_pos=None, init_yaw=None, keep_state=False):
        # type: (Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor, bool) -> Tensor
        history_length = input.shape[0]
        batch_size = input.shape[1]
        n_vehicles = input.shape[2] - 1

        input_ego = input[:, :, :1, :]
        input = input[:, :, 1:, :]

        history_length = history_length - 2 * (self.kernel_size // 2)
        h_ego = (self.conv_input(input_ego.permute(1, 3, 2, 0)).permute(3, 0, 2, 1)
                    .contiguous()
                    .view(history_length, batch_size, 1, self.feature_size))
        h = (self.conv_input(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1)
                            .contiguous()
                            .view(history_length, batch_size, n_vehicles, self.feature_size))
        self._init_state(batch_size, n_vehicles)

        current_pos_ego = init_pos[:, :, :1, :]
        current_pos = init_pos[:, :, 1:, :]

        embedded_lanes = self.lane_embedding(lane_input, keep_state)

        for i in range(history_length):
            current_yaw_ego = input_ego[i:i+1, :, :, 1:2]
            current_yaw = input[i:i+1, :, :, 1:2]
            current_pos_ego = current_pos_ego + self._to_xy(input_ego[i:i+1], current_yaw_ego, 3)
            current_pos = current_pos + self._to_xy(input[i:i+1], current_yaw, 3)
            self.hx_ego, self.cx_ego, self.hx, self.cx = self.attention_lstm(
                h_ego[i:i + 1], current_pos_ego.unsqueeze(3), (self.hx_ego, self.cx_ego),
                h[i:i + 1], current_pos.unsqueeze(3), (self.hx, self.cx),
                mask_input, embedded_lanes,
                lane_mask, batch_size, n_vehicles)



        current_pos_ego = current_pos_ego.unsqueeze(3).repeat(1, 1, 1, self.num_preds, 1)
        current_pos = current_pos.unsqueeze(3).repeat(1, 1, 1, self.num_preds, 1)
        # current_yaw_ego = current_yaw_ego.unsqueeze(3).repeat(1, 1, 1, self.num_preds, 1)
        # current_yaw = current_yaw.unsqueeze(3).repeat(1, 1, 1, self.num_preds, 1)
        pred_sequence_ego = []
        pred_sequence = []
        for i in range(len_pred):
            self.hx_ego, self.cx_ego, self.hx, self.cx = self.attention_lstm(
                self.hx_ego.view((1, batch_size, 1, self.feature_size)), current_pos_ego,
                (self.hx_ego, self.cx_ego),
                self.hx.view((1, batch_size, n_vehicles, self.feature_size)), current_pos, (self.hx, self.cx),
                mask_input, embedded_lanes,
                lane_mask, batch_size, n_vehicles)
            current_output_ego = self.output_layer_ego(self.hx_ego.view((1, batch_size, 1, self.feature_size)), mask_input)
            current_output = self.output_layer(self.hx.view((1, batch_size, n_vehicles, self.feature_size)), mask_input)
            current_yaw_ego = current_output_ego[:, :, :, :, 1:2]
            current_yaw = current_output[:, :, :, :, 1:2]
            current_pos_ego = current_pos_ego + self._to_xy(current_output_ego, current_yaw_ego, 4)
            current_pos = current_pos + self._to_xy(current_output, current_yaw, 4)
            pred_sequence_ego.append(torch.cat((current_pos_ego, current_output_ego.narrow(4, 2, 4)), dim=-1))
            pred_sequence.append(torch.cat((current_pos, current_output.narrow(4, 2, 4)), dim=-1))


        h_ego = torch.cat(pred_sequence_ego, dim=0)
        h = torch.cat(pred_sequence, dim=0)

        return torch.cat((h_ego, h), dim=2)


