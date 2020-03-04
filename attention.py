import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from activation import Mish
import torch.jit as jit


@torch.jit.script
def attention(query, key, value, mask, scale=0):
    # type: (Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    "Compute 'Scaled Dot Product Attention'"
    if scale == 0:
        d_k = query.size(-1)
    else:
        d_k = scale
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if mask is not None:
    #     mask = torch.any(mask, 0)
    #     if mask.ndim == scores.ndim:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     else:
    mask = torch.any(mask, 0)
    scores = scores.masked_fill(mask.unsqueeze(-2) == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


@torch.jit.script
def geometric_attention(query, key, value, mask, scale=0):
    # type: (Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    "Compute 'Scaled Dot Product Attention'"
    if scale == 0:
        d_k = query.size(-1)
    else:
        d_k = scale
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # if mask is not None:
    #     mask = torch.any(mask, 0)
    #     if mask.ndim == scores.ndim:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     else:
    mask = torch.any(mask, 0)
    scores = scores.masked_fill(mask.unsqueeze(-2) == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    # output = torch.exp(torch.matmul(p_attn, torch.log(torch.relu(value)+1e-6)))
    output = torch.exp(torch.matmul(p_attn, torch.log(torch.sigmoid(value))))
    # output = 1/torch.matmul(p_attn, 1/(torch.relu(value)+1e-4))
    return output


@torch.jit.script
def relational_attention(relation, query, key, value, mask, scale=0):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    att = attention(query, key, value, mask, scale)
    return relation*att

@torch.jit.script
def geometric_relational_attention(relation, query, key, value, mask, scale=0):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, int) -> Tensor
    att = geometric_attention(query, key, value, mask, scale)
    return relation*att


class LaneAttention(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads):
        super(LaneAttention, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size

        conv_time_kernel = 1
        padding = 0

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)

        self.query = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=False)

        self.attention_combine = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                           padding=(0, padding), bias=False)

    def forward(self, vehicles, lanes, mask_lanes):
        batch_size = vehicles.shape[1]
        n_vehicles = vehicles.shape[2]
        len_pred = vehicles.shape[0]
        n_lanes = lanes.shape[2]
        n_heads_features = self.feature_size // self.n_heads

        key = self.key_lanes(lanes).view(1, batch_size * n_vehicles, n_lanes, self.n_heads, n_heads_features)
        value = self.value_lanes(lanes).view(1, batch_size * n_vehicles, n_lanes, self.n_heads, n_heads_features)

        query = self.query(vehicles.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).contiguous().view(len_pred, batch_size * n_vehicles,
                                                                                               self.n_heads, 1, n_heads_features)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        mask_lanes = torch.any(mask_lanes, 0).view(1, batch_size * n_vehicles, 1, n_lanes)
        mask_lanes = mask_lanes.repeat((len_pred, 1, self.n_heads, 1))

        output = attention(query, key, value, mask_lanes).reshape(len_pred, batch_size,
                                                                  n_vehicles,
                                                                  self.feature_size)
        output = output.permute(1, 3, 2, 0)

        output = self.attention_combine(output).permute(3, 0, 2, 1) + vehicles

        return output


class SocialAttention(nn.Module):
    def __init__(self, feature_size, n_heads):
        super(SocialAttention, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size

        conv_time_kernel = 1
        padding = 0

        self.key = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                             padding=(0, padding), bias=False)
        self.query = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=False)
        self.value = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=False)

        self.attention_combine = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                           padding=(0, padding), bias=False)

    def forward(self, input, mask):
        batch_size = input.shape[1]
        len_pred = input.shape[0]
        n_vehicles = input.shape[2]
        n_heads_features = self.feature_size // self.n_heads

        key = self.key(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                           n_vehicles, self.n_heads, n_heads_features)
        query = self.query(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                               n_vehicles, self.n_heads, n_heads_features)
        value = self.value(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                               n_vehicles, self.n_heads, n_heads_features)
        key = key.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        output = attention(query, key, value, mask).permute(0, 1, 3, 2, 4).reshape(len_pred, batch_size,
                                                                                                   n_vehicles,
                                                                                                   self.feature_size)
        output = output.permute(1, 3, 2, 0)

        output = self.attention_combine(output).permute(3, 0, 2, 1) + input

        return output.view(len_pred, batch_size * n_vehicles, self.feature_size)


class LaneRelationalAttention(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads):
        super(LaneRelationalAttention, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size

        conv_time_kernel = 1
        padding = 0
        use_bias = True

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=use_bias)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=use_bias)

        self.relation_lanes = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=use_bias)
        self.query = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=use_bias)

        self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=use_bias)

    def forward(self, vehicles, initial_pos, lanes, mask_lanes):
        # type: (Tensor, float, Tensor, Tensor) -> Tensor
        vehicles = vehicles + initial_pos
        batch_size = vehicles.shape[1]
        n_vehicles = vehicles.shape[2]
        len_pred = vehicles.shape[0]
        n_lanes = lanes.shape[1]
        n_heads_features = self.feature_size // self.n_heads

        key = (self.key_lanes(lanes)).view(1, batch_size, n_lanes, self.n_heads, n_heads_features).permute(0, 1, 3, 2, 4)
        value = (self.value_lanes(lanes)).view(1, batch_size, n_lanes, self.n_heads, n_heads_features).permute(0, 1, 3, 2, 4)
        query = (self.query(vehicles.permute(1, 3, 2, 0)).permute(3, 0, 2, 1)
                     .contiguous()
                     .view(len_pred, batch_size, n_vehicles,
                           self.n_heads, n_heads_features)).permute(0, 1, 3, 2, 4)
        relation = (self.relation_lanes(vehicles.permute(1, 3, 2, 0)).permute(3, 0, 2, 1)
                        .contiguous()
                        .view(len_pred, batch_size, n_vehicles,
                              self.n_heads, n_heads_features)).permute(0, 1, 3, 2, 4)
        mask_lanes = torch.any(mask_lanes, 0).view(1, batch_size, 1, n_lanes)
        mask_lanes = mask_lanes.repeat((len_pred, 1, self.n_heads, 1))

        output = geometric_relational_attention(relation, query, key, value, mask_lanes)
        output = output.reshape(len_pred, batch_size, n_vehicles, self.feature_size)

        output = self.attention_combine(output) + vehicles - 2*initial_pos

        return output


class SocialRelationalAttention(nn.Module):
    def __init__(self, feature_size, n_heads):
        super(SocialRelationalAttention, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        use_bias = True

        conv_time_kernel = 1
        padding = 0

        self.key = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                             padding=(0, padding), bias=use_bias)
        self.query = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=use_bias)
        self.value = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                               padding=(0, padding), bias=use_bias)
        self.relation = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                  padding=(0, padding), bias=use_bias)

        self.attention_combine = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                           padding=(0, padding), bias=use_bias)

    def forward(self, input, initial_pos, mask):
        # type: (Tensor, float, Tensor) -> Tensor
        input = input + initial_pos
        batch_size = input.shape[1]
        len_pred = input.shape[0]
        n_vehicles = input.shape[2]
        n_heads_features = self.feature_size // self.n_heads

        key = self.key(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                           n_vehicles, self.n_heads,
                                                                           n_heads_features)
        query = self.query(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                               n_vehicles, self.n_heads,
                                                                               n_heads_features)
        value = self.value(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                               n_vehicles, self.n_heads,
                                                                               n_heads_features)
        relation = self.relation(input.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                                     n_vehicles, self.n_heads,
                                                                                     n_heads_features)

        key = key.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        relation = relation.permute(0, 1, 3, 2, 4)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        output = geometric_relational_attention(
            relation, query, key, value, mask
        ).permute(0, 1, 3, 2, 4).reshape(len_pred, batch_size,
                                         n_vehicles,
                                         self.feature_size)
        output = output.permute(1, 3, 2, 0)

        output = self.attention_combine(output).permute(3, 0, 2, 1) + input - 2*initial_pos

        return output.view(len_pred, batch_size * n_vehicles, self.feature_size)


class LaneFollow(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads):
        super(LaneFollow, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size

        conv_time_kernel = 1
        padding = 0
        use_bias = True

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=use_bias)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=use_bias)
        self.activation = Mish

        if self.n_heads>1:
            self.attention_combine = nn.Linear(self.feature_size, self.feature_size, bias=use_bias)

    def forward(self, query, lanes, mask_input, mask_lanes):
        batch_size = lanes.shape[1]
        n_vehicles = query.shape[2]
        len_pred = query.shape[0]
        n_lanes = lanes.shape[2]
        lane_size = lanes.shape[0]
        n_heads_features = self.feature_size // self.n_heads

        lanes = lanes.permute(1, 2, 0, 3).reshape(1, batch_size, n_lanes*lane_size, self.lane_feature_size)
        value = self.value_lanes(lanes).reshape(1, batch_size, n_lanes*lane_size, self.n_heads, n_heads_features)
        value = value.permute(0, 1, 3, 2, 4)
        mask_lanes = mask_lanes[:lane_size, :, :].permute(1, 0, 2).reshape(1, 1, batch_size, 1, 1, n_lanes*lane_size)
        mask_input = mask_input.reshape(mask_input.shape[0], 1, batch_size, 1, mask_input.shape[2], 1)
        mask = mask_lanes & mask_input
        key = self.key_lanes(lanes)
        key = key.reshape(1, batch_size, n_lanes*lane_size, self.n_heads, n_heads_features)
        key = key.permute(0, 1, 3, 2, 4)

        query = (query
                 .reshape(len_pred, batch_size, n_vehicles, self.n_heads, n_heads_features)
                 .permute(0, 1, 3, 2, 4))

        output = geometric_attention(query, key, value, mask, scale=1)
        output = output.permute(0, 1, 3, 2, 4).reshape(len_pred, batch_size, n_vehicles, self.feature_size)
        if self.n_heads > 1:
            output = self.attention_combine(output)

        return output




