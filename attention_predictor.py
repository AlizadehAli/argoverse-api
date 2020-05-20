import torch
import torch.nn as nn
import torch.functional as F
import torch.jit as jit
import math

# from attention import attention, LaneAttention, SocialAttention
from activation import Mish
from loss_functions import xytheta_activation

import os
from pydoc import locate
from attention import LaneRelationalAttention, SocialRelationalAttention, \
    geometric_relational_attention, relational_attention, geometric_attention

# current_file = os.path.dirname(__file__)
# current_path = os.getcwd()
#
# current_file = current_file.replace(current_path, '').replace("/", ".")[1:]
# LaneRelationalAttention = locate(current_file + ".attention.LaneRelationalAttention")
# SocialRelationalAttention = locate(current_file + ".attention.SocialRelationalAttention")
# geometric_relational_attention = locate(current_file + ".attention.geometric_relational_attention")
# relational_attention = locate(current_file + ".attention.relational_attention")
# geometric_attention = locate(current_file + ".attention.geometric_attention")


class ParametersAttentionPredictor():
    def __init__(self):
        self.feature_size = 60
        self.lane_feature_size = 60
        self.n_heads_past = 6
        self.n_heads_fut = 6
        self.n_heads_lanes = 6
        self.num_preds = 6
        self.n_lane_layers = 2
        self.n_layers = 2
        self.n_loop = 1

class Embedding(nn.Module):

    def __init__(self, feature_size, n_layers=1):
        super(Embedding, self).__init__()
        self.feature_size = feature_size
        self.n_layers = n_layers
        self.kernel_size = 3
        self.conv_input = nn.Conv2d(2, self.feature_size, (1, self.kernel_size), padding=(0, 0), stride=(1, 1))
        self.LSTM = nn.LSTM(self.feature_size, self.feature_size, num_layers=n_layers)

        self.hx = None
        self.cx = None

    def forward(self, input, keep_state=False):
        history_length = input.shape[0]
        batch_size = input.shape[1]
        n_vehicles = input.shape[2]
        if not keep_state:
            if torch.cuda.is_available():
                self.hx = torch.zeros(self.n_layers, batch_size*n_vehicles, self.feature_size).cuda()
                self.cx = torch.zeros(self.n_layers, batch_size*n_vehicles, self.feature_size).cuda()
            else:
                self.hx = torch.zeros(self.n_layers, batch_size * n_vehicles, self.feature_size)
                self.cx = torch.zeros(self.n_layers, batch_size * n_vehicles, self.feature_size)
        features = (self.conv_input(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1)
                    .contiguous()
                    .view(history_length - 2*(self.kernel_size//2), batch_size*n_vehicles, self.feature_size))
        features = torch.tanh(features)
        _, (self.hx, self.cx) = self.LSTM(features, (self.hx, self.cx))
        return self.hx[-1].view(batch_size, n_vehicles, self.feature_size), (self.hx, self.cx)

class LaneEmbedding(nn.Module):
    def __init__(self, feature_size, n_layers):
        super(LaneEmbedding, self).__init__()
        self.feature_size = feature_size // 2

        self.n_layers = n_layers
        self.kernel_size = 3
        self.conv_input = nn.Conv2d(2, self.feature_size, (1, self.kernel_size), padding=(0, 0), stride=(1, 1))
        self.LSTM = nn.LSTM(self.feature_size, self.feature_size, num_layers=n_layers, bidirectional=True)

        self.hx = None
        self.cx = None

    def forward(self, input, keep_state=False):
        history_length = input.shape[0]
        batch_size = input.shape[1]
        n_lanes = input.shape[2]
        if not keep_state:
            if torch.cuda.is_available():
                self.hx = torch.zeros(2*self.n_layers, batch_size * n_lanes, self.feature_size).cuda()
                self.cx = torch.zeros(2*self.n_layers, batch_size * n_lanes, self.feature_size).cuda()
            else:
                self.hx = torch.zeros(2*self.n_layers, batch_size * n_lanes, self.feature_size)
                self.cx = torch.zeros(2*self.n_layers, batch_size * n_lanes, self.feature_size)

        features = self.conv_input(
            input.view(history_length, batch_size, n_lanes, -1
                       ).permute(1, 3, 2, 0)).permute(3, 0, 2, 1).contiguous().view(
                       -1, batch_size * n_lanes, self.feature_size)
        # features = torch.tanh(features)
        _, (self.hx, self.cx) = self.LSTM(features, (self.hx, self.cx))
        return self.hx[-2:].permute(1, 0, 2).contiguous().view(batch_size, n_lanes, self.feature_size * 2)


@jit.script
def outputActivation(pos, x, dim):
    # type: (float, Tensor, int) -> Tensor
    # pos = pos.unsqueeze(-2)
    p = x.narrow(dim, 5, 1)
    n_pred = p.shape[dim - 1]
    p = (p/math.sqrt(n_pred)).softmax(dim-1)
    muX = x.narrow(dim, 0, 1)# + pos.narrow(dim, 0, 1)
    muY = x.narrow(dim, 1, 1)# + pos.narrow(dim, 1, 1)
    sigX = x.narrow(dim, 2, 1)
    sigY = x.narrow(dim, 3, 1)
    rho = x.narrow(dim, 4, 1)
    sigX = torch.exp(sigX/2)
    sigY = torch.exp(sigY/2)
    rho = torch.tanh(rho)

    out = torch.cat([muX, muY, sigX, sigY, rho, p], dim=dim)
    return out


class MultiOutputLayer(nn.Module):

    def __init__(self, feature_size, num_preds):
        super(MultiOutputLayer, self).__init__()
        self.feature_size = feature_size
        self.num_preds = num_preds
        self.state_size = 10

        self.layer1 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        self.layer2 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        self.output = nn.Conv2d(self.feature_size, self.state_size * num_preds, (1, 1))

        self.activation = Mish

    def forward(self, pos, inputs, mask):
        inputs = inputs.permute(1, 3, 2, 0)
        inputs_shape = inputs.shape
        h = self.activation(self.layer1(inputs))
        h = self.activation(self.layer2(h))
        # h = inputs
        h = self.output(h).permute(3, 0, 2, 1).view(inputs_shape[3], inputs_shape[0],
                                                    inputs_shape[2], self.num_preds, self.state_size)
        output = xytheta_activation(h, 4)
        return output


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.weight_att = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.bias_att = nn.Parameter(torch.randn(2 * hidden_size))

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor]
        hx, cx, att = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate_att, attgate = (torch.mm(att, self.weight_att.t()) + self.bias_att).chunk(2, 1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        ingate_att = torch.sigmoid(ingate_att)
        attgate = torch.tanh(attgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate) + (ingate_att * attgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class AttentionLSTM(nn.Module):
    def __init__(self, feature_size, n_heads, lane_feature_size, n_heads_lanes):
        super(AttentionLSTM, self).__init__()
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.lane_feature_size = lane_feature_size
        self.n_heads_lanes = n_heads_lanes
        self.cell = jit.script(LSTMCell(self.feature_size, self.feature_size))
        # self.cell = LSTMCell(self.feature_size, self.feature_size)
        # self.lane_attention = LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes)
        self.social_attention = jit.script(SocialRelationalAttention(self.feature_size, self.n_heads))
        # self.layer_norm = nn.LayerNorm(self.feature_size)


    def forward(self, input, state, mask, lane_mask, len_pred, batch_size, n_vehicles):
    # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, int, int, int) -> Tensor
        outputs = torch.jit.annotate(List[Tensor], [])
        # outputs = []
        prev_out = input.view(batch_size*n_vehicles, self.feature_size)
        hx, cx = state
        hx = hx.view(batch_size*n_vehicles, self.feature_size)
        cx = cx.view(batch_size*n_vehicles, self.feature_size)
        for i in range(len_pred):
            # prev_out = self.lane_attention(prev_out.view(1, batch_size, n_vehicles, self.feature_size), lane, lane_mask)
            att = hx.view(1, batch_size, n_vehicles, self.feature_size)
            att = self.social_attention(att, mask).view(batch_size*n_vehicles, self.feature_size)
            hx, cx = self.cell(prev_out, (hx, cx, att))
            prev_out = prev_out + hx
            outputs += [prev_out]
        return torch.stack(outputs)


class AttentionPredictor(nn.Module, ParametersAttentionPredictor):

    def __init__(self, separate_ego=False):
        nn.Module.__init__(self)
        ParametersAttentionPredictor.__init__(self)
        self.separate_ego = separate_ego
        self.traj_embedding = Embedding(self.feature_size, self.n_layers)
        if separate_ego:
            self.ego_embedding = Embedding(self.feature_size, self.n_layers)
        self.pos_embedding = nn.Linear(2, self.feature_size)
        self.lane_embedding = LaneEmbedding(self.lane_feature_size, self.n_lane_layers)

        self.lane_attention = LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes)
        self.social_attention = SocialRelationalAttention(self.feature_size, self.n_heads_past)
        # self.lane_attention =LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes)
        # self.social_attention = SocialRelationalAttention(self.feature_size, self.n_heads_past)
        # self.layer_norm = nn.LayerNorm(self.feature_size)

        self.pred_LSTM = nn.LSTM(self.feature_size, self.feature_size, num_layers=self.n_layers)
        if self.separate_ego:
            self.pred_LSTM_ego = nn.LSTM(self.feature_size, self.feature_size, num_layers=self.n_layers)
        # self.lane_attention_fut = jit.script(LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes))
        # self.social_attention_fut = jit.script(SocialRelationalAttention(self.feature_size, self.n_heads_past))

        if self.n_loop > 0:
            self.reencode_LSTM = nn.LSTM(self.feature_size, self.feature_size, num_layers=self.n_layers)
            self.reencoded_lane_attention = LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes)
            self.reencoded_social_attention = SocialRelationalAttention(self.feature_size, self.n_heads_past)
        # self.lane_attention_fut = LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes)
        # self.social_attention_fut = SocialRelationalAttention(self.feature_size, self.n_heads_past)

        # self.pred_LSTM2 = nn.LSTM(self.feature_size, self.feature_size, num_layers=self.n_layers)
        # self.lane_attention_fut2 = jit.script(LaneRelationalAttention(self.feature_size, self.lane_feature_size, self.n_heads_lanes))
        # self.social_attention_fut2 = jit.script(SocialRelationalAttention(self.feature_size, self.n_heads_past))



        # self.attention_LSTM = jit.script(AttentionLSTM(self.feature_size, self.n_heads_fut,
        #                                                self.lane_feature_size, self.n_heads_lanes))
        # self.attention_LSTM = AttentionLSTM(self.feature_size, self.n_heads_fut,
        #                                                self.lane_feature_size, self.n_heads_lanes)

        # self.layer_norm_fut = nn.LayerNorm(self.feature_size)
        self.output_layer = MultiOutputLayer(self.feature_size, self.num_preds)

        self.hx = None
        self.cx = None
	
    def set_training(self, train):
        self.lane_attention_fut.training = train
        self.social_attention_fut.training = train
        self.lane_attention.training = train
        self.social_attention.training = train

    def get_lane_attention_matrix(self):
        return self.lane_attention.get_attention_matrix()

    def get_social_attention_matrix(self):
        return self.social_attention.get_attention_matrix()

    def forward(self, input, mask_input, lane_input=None, lane_mask=None, len_pred=30, init_pos=None, keep_state=False):
        batch_size = input.shape[1]
        n_vehicles = input.shape[2]

        if self.separate_ego:
            h, (self.hx, self.cx) = self.traj_embedding(input[:, : , 1:, :], keep_state)
            h_ego, (self.hx_ego, self.cx_ego) = self.ego_embedding(input[:, :, 0:1, :], keep_state)
            h_ego = h_ego.unsqueeze(0)
        else:
            h, (self.hx, self.cx) = self.traj_embedding(input, keep_state)

        h = h.unsqueeze(0)
        if init_pos is not None:
            h_pos = torch.tanh(self.pos_embedding(init_pos))
        else:
            h_pos = 0

        is_lane = False
        if lane_input is not None and lane_input.shape[2]>0:
            is_lane = True
            embedded_lanes = self.lane_embedding(lane_input, keep_state)

        if self.separate_ego:
            h = torch.cat((h_ego, h), dim=2)
        if is_lane:
            h = self.lane_attention(h, h_pos, embedded_lanes, mask_input, lane_mask)
        h = self.social_attention(h, h_pos, mask_input)
        # h = self.layer_norm(h)
        h = h.repeat((len_pred, 1, 1))

        if self.separate_ego:
            h = h.view(len_pred, batch_size, n_vehicles, self.feature_size)
            h_ego = h[:, :, 0:1, :].view(len_pred, batch_size, self.feature_size)
            h = h[:, :, 1:, :].reshape(len_pred, batch_size*(n_vehicles-1), self.feature_size)
            h, _ = self.pred_LSTM(h, (self.hx, self.cx))
            h_ego, _ = self.pred_LSTM_ego(h_ego, (self.hx_ego, self.cx_ego))
            h = h.view(len_pred, batch_size, n_vehicles-1, self.feature_size)
            h_ego = h_ego.view(len_pred, batch_size, 1, self.feature_size)
            h = torch.cat((h_ego, h), dim=2)
        else:
            h, _ = self.pred_LSTM(h, (self.hx, self.cx))
            h = h.view(len_pred, batch_size, n_vehicles, self.feature_size)
        # if init_pos is not None:
        #     h = self.lane_attention_fut(torch.cumsum(h, dim=0), h_pos, embedded_lanes, lane_mask)
        #     # h = self.social_attention_fut(torch.cumsum(h, dim=0), h_pos, mask_input)
        # else:
        #     h = self.lane_attention_fut(h, h_pos, embedded_lanes, lane_mask)
        #     # h = self.social_attention_fut(h, h_pos, mask_input)
        #     init_pos = 0

        for i in range(self.n_loop):
            h = h.view(len_pred, batch_size * n_vehicles, self.feature_size)
            _, (h, _) = self.reencode_LSTM(h, (self.hx, self.cx))
            h = h[-1].view(1, batch_size, n_vehicles, self.feature_size)
            if is_lane:
                h = self.reencoded_lane_attention(h, h_pos, embedded_lanes, mask_input, lane_mask)
            h = self.reencoded_social_attention(h, h_pos, mask_input)
            h = h.repeat((len_pred, 1, 1))
            h, _ = self.pred_LSTM(h, (self.hx, self.cx))
            h = h.view(len_pred, batch_size, n_vehicles, self.feature_size)
            # h = self.lane_attention_fut(h, h_pos, embedded_lanes, lane_mask)
            # h = self.social_attention_fut(h, h_pos, mask_input)

        h = h.view(len_pred, batch_size, n_vehicles, self.feature_size)
        # h = self.layer_norm_fut(h)
        h = self.output_layer(init_pos, h, mask_input)

        return h
