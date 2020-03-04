import torch
import torch.nn.functional as F
import torch.nn as nn
import math
# from jit_LSTMCell import script_predlstm
# from torch.autograd.gradcheck import zero_gradients
# import torch.jit as jit

def distances(inputs):
    n_vehicles = inputs.shape[1]
    inputs_tiled = inputs.unsqueeze(2).repeat((1, 1, n_vehicles, 1))
    diff = inputs_tiled - inputs_tiled.permute(0, 2, 1, 3)
    dist = torch.exp(-torch.mean(diff * diff, 3))
    return dist

def distances2(inputs):
    n_vehicles = inputs.shape[2]
    inputs_narrowed = inputs.narrow(4, 0, 2)
    p = inputs.narrow(4, 5, 1).transpose(4, 3)
    inputs_tiled = inputs_narrowed.unsqueeze(3).repeat((1, 1, 1, n_vehicles, 1, 1))
    diff = inputs_tiled - inputs_tiled.permute(0, 1, 3, 2, 4, 5)
    dist = torch.sum(diff * diff, 5)
    min_dist, _ = torch.min(dist, 4)
    mean_dist = torch.sum(dist*p, 4)
    # return (min_dist - mean_dist).detach() + mean_dist
    return min_dist

def distances3(inputs):
    n_vehicles = inputs.shape[2]
    inputs_tiled = inputs.unsqueeze(3).repeat((1, 1, 1, n_vehicles, 1))
    diff = inputs_tiled - inputs_tiled.permute(0, 1, 3, 2, 4)
    dist = torch.sum(diff * diff, 4)
    return dist

def attention_distance(inputs, value, mask):
    "Compute 'Scaled Dot Product Attention'"
    scores = distances(inputs)

    if mask is not None:
        mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
        mask_veh = mask_veh.repeat((int(inputs.shape[0]/mask_veh.shape[0]), 1, 1))
        scores = scores.masked_fill(mask_veh == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = scores
    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = torch.any(mask, 0)
        scores = scores.masked_fill(mask.unsqueeze(2) == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = scores
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


def local_attention(distance, query, key, value, mask):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask_veh = (1 - torch.prod(1-mask, 0)).unsqueeze(2)
        scores = scores.masked_fill(mask_veh == 0, -1e9)
    # print(torch.max(distance))
    scores = scores - distance
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = scores
    # if dropout is not None:
    #     p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


def outputActivation(pos, x, mask, dim):
    # type: (Tensor, int) -> Tensor
    if x.shape[-1] > 5:
        p = x.narrow(dim, 5, 1)
        n_pred = p.shape[dim - 1]
        # mask_veh = (1 - torch.prod(1 - mask, 0, keepdim=True)).view(1, x.shape[1], x.shape[2], 1, 1)
        # p = p.masked_fill(mask_veh == 0, -1e9)
        p = nn.Softmax(dim-1)(p/math.sqrt(n_pred))
    pos = pos.view(1, pos.shape[0], pos.shape[1], 1, pos.shape[2])
    muX = x.narrow(dim, 0, 1) + pos.narrow(dim, 0, 1)
    muY = x.narrow(dim, 1, 1) + pos.narrow(dim, 1, 1)
    sigX = x.narrow(dim, 2, 1)
    sigY = x.narrow(dim, 3, 1)
    rho = x.narrow(dim, 4, 1)
    sigX = torch.exp(sigX/2)
    sigY = torch.exp(sigY/2)
    rho = torch.tanh(rho)

    out = torch.cat([muX, muY, sigX, sigY, rho, p], dim=dim)
    return out


class SelfAttentionFut(nn.Module):
    def __init__(self, feature_size, n_heads, output_layer=None):
        super(SelfAttentionFut, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        self.output_layer = output_layer

        conv_time_kernel = 1
        padding = 0

        self.key_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                 padding=(0, padding), bias=False)
        self.query_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                   padding=(0, padding), bias=False)
        self.value_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                   padding=(0, padding), bias=False)

        self.attention_combine_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                               padding=(0, padding), bias=False)

    def forward(self, inputs, mask=None):
        batch_size = inputs.shape[0]
        len_pred = inputs.shape[3]
        n_vehicles = inputs.shape[2]
        n_heads_features = int(self.feature_size / self.n_heads)

        # output_preds = self.output_layer(pos,
        #                                  inputs,
        #                                  mask)  # .permute(3, 0, 2, 1).view(len_pred, batch_size, n_vehicles, self.num_preds, 6)

        key = self.key_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                            n_vehicles, self.n_heads, n_heads_features)
        query = self.query_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                n_vehicles, self.n_heads, n_heads_features)
        value = self.value_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                n_vehicles, self.n_heads, n_heads_features)
        key = key.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        if mask is not None:
            mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        # distance = distances2(output_preds).unsqueeze(2).repeat((1, 1, self.n_heads, 1, 1))
        dropout = None
        output = attention(query, key, value, mask, dropout).permute(0, 1, 3, 2, 4).reshape(len_pred, batch_size,
                                                                                                   n_vehicles,
                                                                                                   self.feature_size)

        output = output.permute(1, 3, 2, 0)

        output = self.attention_combine_fut(output) + inputs

        return output

    def get_attention_matrices(self, pos, inputs, mask):
        batch_size = inputs.shape[0]
        len_pred = inputs.shape[3]
        n_vehicles = inputs.shape[2]
        n_heads_features = int(self.feature_size / self.n_heads)

        output_preds = self.output_layer(pos, inputs,
                                         mask)  # .permute(3, 0, 2, 1).view(len_pred, batch_size, n_vehicles, self.num_preds, 6)
        # output_preds = outputActivation(output_preds, mask, dim=4)

        key = self.key_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                            n_vehicles, self.n_heads, n_heads_features)
        query = self.query_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                n_vehicles, self.n_heads, n_heads_features)

        key = key.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)
        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        distance = distances2(output_preds).unsqueeze(2).repeat((1, 1, self.n_heads, 1, 1))

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
            scores = scores.masked_fill(mask_veh == 0, -1e9)
        # print(torch.max(distance))
        scores = scores  # - distance
        p_attn = F.softmax(scores, dim=-1)

        return p_attn, output_preds, distance
    
    
class LaneAttentionFut(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads, output_layer=None):
        super(LaneAttentionFut, self).__init__()

        self.n_heads = n_heads
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size
        self.output_layer = output_layer

        conv_time_kernel = 1
        padding = 0

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)

        self.query_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                   padding=(0, padding), bias=False)

        self.attention_combine_fut = nn.Conv2d(self.feature_size, self.feature_size, (1, conv_time_kernel),
                                               padding=(0, padding), bias=False)

    def forward(self, inputs, lanes):
        batch_size = inputs.shape[0]
        len_pred = inputs.shape[3]
        n_vehicles = inputs.shape[2]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_lanes(lanes).view(1, batch_size, n_lanes, self.n_heads, n_heads_features)
        value = self.value_lanes(lanes).view(1, batch_size, n_lanes, self.n_heads, n_heads_features)

        query = self.query_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                n_vehicles, self.n_heads, n_heads_features)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)


        # distance = distances2(output_preds).unsqueeze(2).repeat((1, 1, self.n_heads, 1, 1))
        dropout = None
        output = attention(query, key, value, None, dropout).permute(0, 1, 3, 2, 4).reshape(len_pred, batch_size,
                                                                                                   n_vehicles,
                                                                                                   self.feature_size)

        output = output.permute(1, 3, 2, 0)

        output = self.attention_combine_fut(output) + inputs

        return output

    def get_attention_matrices(self, pos, inputs, lanes):
        batch_size = inputs.shape[0]
        len_pred = inputs.shape[3]
        n_vehicles = inputs.shape[2]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        output_preds = self.output_layer(pos, inputs)  # .permute(3, 0, 2, 1).view(len_pred, batch_size, n_vehicles, self.num_preds, 6)
        # output_preds = outputActivation(output_preds, mask, dim=4)

        key = self.key_lanes(lanes).view(1, batch_size, n_lanes, self.n_heads, n_heads_features)
        query = self.query_fut(inputs).permute(3, 0, 2, 1).view(len_pred, batch_size,
                                                                n_vehicles, self.n_heads, n_heads_features)

        key = key.permute(0, 1, 3, 2, 4)
        query = query.permute(0, 1, 3, 2, 4)

        distance = distances2(output_preds).unsqueeze(2).repeat((1, 1, self.n_heads, 1, 1))

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        # print(torch.max(distance))
        scores = scores  # - distance
        p_attn = F.softmax(scores, dim=-1)

        return p_attn, output_preds, distance


class SelfAttentionPast(nn.Module):
    def __init__(self, feature_size, n_heads):
        super(SelfAttentionPast, self).__init__()
        self.feature_size = feature_size
        self.n_heads = n_heads
        self.fake_vehicles = 0
        n_heads_features = int(self.feature_size / self.n_heads)

        self.key_past = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_past = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.value_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

        self.keys_mem = nn.Parameter(torch.randn(1, self.fake_vehicles, self.n_heads, n_heads_features))
        self.memory = nn.Parameter(torch.randn(1, self.fake_vehicles, self.n_heads, n_heads_features) * 1e-3)

        self.attention_combine_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

    def forward(self, pos, inputs, mask):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        key_mem_tiled = self.keys_mem.repeat((batch_size, 1, 1, 1))
        key = torch.cat((key, key_mem_tiled), dim=1)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        value = self.value_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        memory_tiled = self.memory.repeat((batch_size, 1, 1, 1))
        value = torch.cat((value, memory_tiled), dim=1)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))
        # pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        # diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        # distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        dropout = None
        output = attention(query, key, value, mask, dropout).permute(0, 2, 1, 3).reshape(batch_size * n_vehicles,
                                                                                                self.feature_size)

        output = self.attention_combine_past(output) + inputs.view(batch_size*n_vehicles, self.feature_size)

        return output

    def get_attention_matrices(self, pos, inputs, mask):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        key_mem_tiled = self.keys_mem.repeat((batch_size, 1, 1, 1))
        key = torch.cat((key, key_mem_tiled), dim=1)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
            scores[:, :, :, :mask_veh.shape[-1]] = scores[:, :, :, :mask_veh.shape[-1]].masked_fill(mask_veh == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        return p_attn, distance


class SelfAttentionPastLanes(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads):
        super(SelfAttentionPastLanes, self).__init__()
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size
        self.n_heads = n_heads
        n_heads_features = int(self.feature_size / self.n_heads)

        self.key_past = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.query_past = nn.Linear(self.feature_size, self.feature_size, bias=False)
        self.value_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)

        self.attention_combine_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

    def forward(self, pos, inputs, lanes, mask):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        key_lane = self.key_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)
        key = torch.cat((key, key_lane), dim=1)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        value = self.value_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        value_lane = self.value_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)
        value = torch.cat((value, value_lane), dim=1)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))
        # pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        # diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        # distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        dropout = None
        output = attention(query, key, value, mask, dropout).permute(0, 2, 1, 3).reshape(batch_size * n_vehicles,
                                                                                                self.feature_size)

        output = self.attention_combine_past(output) + inputs.view(batch_size*n_vehicles, self.feature_size)

        return output

    def get_attention_matrices(self, pos, inputs, lanes, mask):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        key_lane = self.key_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)
        key = torch.cat((key, key_lane), dim=1)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
            scores[:, :, :, :mask_veh.shape[-1]] = scores[:, :, :, :mask_veh.shape[-1]].masked_fill(mask_veh == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        return p_attn, distance


class LaneAttentionPast(nn.Module):
    def __init__(self, feature_size, lane_feature_size, n_heads):
        super(LaneAttentionPast, self).__init__()
        self.feature_size = feature_size
        self.lane_feature_size = lane_feature_size
        self.n_heads = n_heads
        n_heads_features = int(self.feature_size / self.n_heads)

        self.query_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

        self.key_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)
        self.value_lanes = nn.Linear(self.lane_feature_size, self.feature_size, bias=False)

        self.attention_combine_past = nn.Linear(self.feature_size, self.feature_size, bias=False)

    def forward(self, pos, inputs, lanes):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)
        value = self.value_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        # pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        # diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        # distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        dropout = None
        output = attention(query, key, value, None, dropout).permute(0, 2, 1, 3).reshape(batch_size * n_vehicles,
                                                                                                self.feature_size)

        output = self.attention_combine_past(output) + inputs.view(batch_size*n_vehicles, self.feature_size)

        return output

    def get_attention_matrices(self, pos, inputs, lanes, mask):
        batch_size = inputs.shape[0]
        n_vehicles = inputs.shape[1]
        n_lanes = lanes.shape[1]
        n_heads_features = int(self.feature_size / self.n_heads)

        key = self.key_lanes(lanes).view(batch_size, n_lanes, self.n_heads, n_heads_features)
        query = self.query_past(inputs).view(batch_size, n_vehicles, self.n_heads, n_heads_features)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        mask = mask.unsqueeze(2).repeat((1, 1, self.n_heads, 1))

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            mask_veh = (1 - torch.prod(1 - mask, 0)).unsqueeze(2)
            scores[:, :, :, :mask_veh.shape[-1]] = scores[:, :, :, :mask_veh.shape[-1]].masked_fill(mask_veh == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)

        pos_tiled = pos.unsqueeze(2).repeat(1, 1, n_vehicles, 1)
        diff = pos_tiled - pos_tiled.permute(0, 2, 1, 3)
        distance = torch.sum(diff * diff, 3).unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        return p_attn, distance


class MultiOutputLayer(nn.Module):

    def __init__(self, feature_size, num_preds):
        super(MultiOutputLayer, self).__init__()
        self.feature_size = feature_size
        self.num_preds = num_preds

        self.layer1 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        self.layer2 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        self.output = nn.Conv2d(self.feature_size, 6 * num_preds, (1, 1))

    def forward(self, pos, inputs, mask):
        inputs_shape = inputs.shape
        h = nn.ReLU()(self.layer1(inputs))
        h = nn.ReLU()(self.layer2(h))
        h = self.output(h).permute(3, 0, 2, 1).view(inputs_shape[3], inputs_shape[0],
                                                    inputs_shape[2], self.num_preds, 6)
        output = outputActivation(pos, h, mask, 4)
        return output


class Embedding(nn.Module):
    def __init__(self, feature_size, n_head):
        super(Embedding, self).__init__()
        self.feature_size = feature_size
        self.n_head = n_head
        self.min_dist = 15

        self.layer = nn.Conv2d(2, self.feature_size, (1, 3), padding=(0, 1), stride=(1, 1))
        # self.layer1 = nn.Conv2d(2, int(self.feature_size/4), (1, 3), padding=(0, 1), stride=(1, 1))
        # self.layer2 = nn.Conv2d(int(self.feature_size/4), int(self.feature_size/2), (1, 3), padding=(0, 1), stride=(1, 2))
        # self.attention1 = SelfAttentionFut(int(self.feature_size/2), self.n_head)
        # self.layer3 = nn.Conv2d(int(self.feature_size/2), self.feature_size, (1, 3), padding=(0, 1), stride=(1, 2))
        # self.attention2 = SelfAttentionFut(self.feature_size, self.n_head)

    def forward(self, inputs):
        # eps = 1e-3
        # n_vehicles = inputs.shape[2]
        # hist = inputs.permute((3, 0, 2, 1))
        # hist_tiled = hist.unsqueeze(3).repeat((1, 1, 1, n_vehicles, 1))
        # diff = hist_tiled - hist_tiled.permute(0, 1, 3, 2, 4)
        # dist = torch.sum(diff * diff, 4)
        #
        # if torch.cuda.is_available():
        #     I = torch.eye(n_vehicles).cuda()
        # else:
        #     I = torch.eye(n_vehicles)
        # # I = I.view(1, 1, n_vehicles, n_vehicles)
        # A = (dist > self.min_dist*self.min_dist).float()
        # A = (A + eps) / torch.sum(A + eps, dim=3, keepdims=True) + I
        #
        # h = self.layer1(inputs)
        # h = torch.matmul(A, h.permute(3, 0, 2, 1)).permute(1, 3, 2, 0)
        # h = self.layer2(h)
        # A = A[::2]
        # h = torch.matmul(A, h.permute(3, 0, 2, 1)).permute(1, 3, 2, 0)
        # h = self.layer3(h)
        # A = A[::2]
        # output = torch.matmul(A, h.permute(3, 0, 2, 1)).permute(1, 3, 2, 0)
        output = self.layer(inputs)
        return output

    def output_time_len(self, input_time_len):
        # return input_time_len // 4
        return input_time_len


class DumbPredictor(nn.Module):

    def __init__(self):
        super(DumbPredictor, self).__init__()

        self.feature_size = 128
        self.lane_feature_size = 32
        self.n_head_past = 8
        self.n_head_fut = 4
        self.num_layers = 1
        self.num_preds = 6

        # self.input_embedding = nn.Conv2d(2, self.feature_size, (1, 3), padding=(0, 0))
        self.input_embedding = Embedding(self.feature_size, self.n_head_past)
        self.lane_embedding = Embedding(self.lane_feature_size, self.n_head_past)

        self.past_LSTM = nn.LSTM(self.feature_size, self.feature_size, self.num_layers, )
        self.lane_LSTM = nn.LSTM(self.lane_feature_size, self.lane_feature_size, self.num_layers, bidirectional=True)

        # self.past_attention = SelfAttentionPast(self.feature_size, self.n_head_past)
        self.lane_attention = LaneAttentionPast(self.feature_size, self.lane_feature_size, self.n_head_past)
        self.past_attention = SelfAttentionPast(self.feature_size, self.n_head_past)
        # self.linear_1 = nn.Linear(self.feature_size, self.feature_size)
        # self.past_attention2 = SelfAttentionPast(self.feature_size, self.n_head_past)
        # self.linear_2 = nn.Linear(self.feature_size, self.feature_size)
        # self.past_attention3 = SelfAttentionPast(self.feature_size, self.n_head_past)

        self.fut_LSTM = nn.LSTM(self.feature_size, self.feature_size, self.num_layers)

        self.output_layer = MultiOutputLayer(self.feature_size, self.num_preds)

        self.fut_attention1 = SelfAttentionFut(self.feature_size, self.n_head_fut, self.output_layer)
        self.fut_lane_attention = LaneAttentionFut(self.feature_size, self.lane_feature_size, self.n_head_past)

        self.pred_LSTM = nn.LSTM(self.feature_size, self.feature_size, self.num_layers, bidirectional=True)
        # self.conv_layer1 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        # self.fut_attention2 = SelfAttentionFut(self.feature_size, self.n_head_fut, self.output_layer)
        # self.conv_layer2 = nn.Conv2d(self.feature_size, self.feature_size, (1, 1))
        # self.fut_attention3 = SelfAttentionFut(self.feature_size, self.n_head_fut, self.output_layer)

    def forward(self, hist, lanes, len_pred, mask, is_pretrain):

        history_length = self.input_embedding.output_time_len(hist.shape[0])
        lanes_length = self.input_embedding.output_time_len(lanes.shape[0])
        batch_size = hist.shape[1]
        n_vehicles = hist.shape[2]
        n_lanes = lanes.shape[2]

        pos = hist[-1]

        embedded = self.input_embedding(hist.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).contiguous().view(history_length, batch_size*n_vehicles, self.feature_size)
        embedded_lanes = self.lane_embedding(lanes.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).contiguous().view(lanes_length, batch_size*n_lanes, self.lane_feature_size)

        if torch.cuda.is_available():
            hx = torch.zeros(self.num_layers, batch_size*n_vehicles, self.feature_size).cuda()
            cx = torch.zeros(self.num_layers, batch_size*n_vehicles, self.feature_size).cuda()
            hx_lanes = torch.zeros(self.num_layers*2, batch_size*n_lanes, self.lane_feature_size).cuda()
            cx_lanes = torch.zeros(self.num_layers*2, batch_size*n_lanes, self.lane_feature_size).cuda()
        else:
            hx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size)
            cx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size)
            hx_lanes = torch.zeros(self.num_layers*2, batch_size * n_lanes, self.lane_feature_size)
            cx_lanes = torch.zeros(self.num_layers*2, batch_size * n_lanes, self.lane_feature_size)

        _, (hx, cx) = self.past_LSTM(embedded, (hx, cx))
        _, (hx_lanes, cx_lanes) = self.lane_LSTM(embedded_lanes, (hx_lanes, cx_lanes))
        embedded_past = torch.sum(hx, 0).view(batch_size, n_vehicles, self.feature_size)
        embedded_lanes = torch.sum(hx_lanes, 0).view(batch_size, n_lanes, self.lane_feature_size)

        lane_attention = self.lane_attention(hist[-1],
                                             embedded_past,
                                             embedded_lanes
                                             )

        past_transformed = self.past_attention(hist[-1],
                                               lane_attention.view(batch_size, n_vehicles, self.feature_size),
                                               mask)
        # past_transformed = nn.ReLU()(self.linear_1(past_transformed))
        # past_transformed = self.past_attention2(hist[-1],
        #                                        past_transformed.view(batch_size, n_vehicles, self.feature_size),
        #                                        mask)
        # past_transformed = nn.ReLU()(self.linear_2(past_transformed))
        # past_transformed = self.past_attention3(hist[-1],
        #                                        past_transformed.view(batch_size, n_vehicles, self.feature_size),
        #                                        mask)

        past_transformed_tiled = past_transformed.unsqueeze(0).repeat((len_pred, 1, 1))

        # fut, _ = self.fut_LSTM(past_transformed, (hx, cx), len_pred)
        fut, (hx2, cx2) = self.fut_LSTM(past_transformed_tiled, (hx, cx))

        fut = fut.permute(1, 2, 0).reshape(batch_size, n_vehicles, self.feature_size, len_pred).permute(0, 2, 1, 3)

        # fut = self._transformer_fut(fut, mask)
        fut = self.fut_lane_attention(fut, embedded_lanes)
        fut = self.fut_attention1(fut, mask)

        fut, _ = self.pred_LSTM(fut.permute(3, 0, 1, 2).reshape(len_pred, batch_size*n_vehicles, self.feature_size), ([hx, hx2], [cx, cx2]))
        fut = fut.reshape(len_pred, batch_size, n_vehicles, self.feature_size).permute(1, 2, 3, 0)

        # fut = nn.ReLU()(self.conv_layer1(fut))
        # fut = self.fut_attention2(fut, mask)
        # fut = nn.ReLU()(self.conv_layer2(fut))
        # fut = self.fut_attention3(fut, mask)

        pred_out = self.output_layer(pos, fut, mask)#.permute(3, 0, 2, 1).view(len_pred, batch_size, n_vehicles, self.num_preds, 6)

        # pred_out = outputActivation(all_preds, mask, 4)

        # output = torch.cat([mu_out, sig_x, sig_y, rho], 3)

        return pred_out

    def get_attention_matrices(self, hist, len_pred, mask):
        history_length = hist.shape[0] - 2
        batch_size = hist.shape[1]
        n_vehicles = hist.shape[2]
        pos = hist[-1]

        embedded = self.input_embedding(hist.permute(1, 3, 2, 0)).permute(3, 0, 2, 1).contiguous().view(history_length,
                                                                                                        batch_size * n_vehicles,
                                                                                                        self.feature_size)

        if torch.cuda.is_available():
            hx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size).cuda()
            cx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size).cuda()
        else:
            hx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size)
            cx = torch.zeros(self.num_layers, batch_size * n_vehicles, self.feature_size)

        _, (hx, cx) = self.past_LSTM(embedded, (hx, cx))

        attention_past, distance_past = self.past_attention.get_attention_matrices(hist[-1],
                                                            torch.sum(hx, 0).view(batch_size, n_vehicles, self.feature_size),
                                                            mask)

        past_transformed = self.past_attention(hist[-1],
                                               torch.sum(hx, 0).view(batch_size, n_vehicles, self.feature_size),
                                               mask)

        past_transformed_tiled = past_transformed.unsqueeze(0).repeat((len_pred, 1, 1))

        # fut, _ = self.fut_LSTM(past_transformed, (hx, cx), len_pred)
        fut, _ = self.fut_LSTM(past_transformed_tiled, (hx, cx))

        fut = fut.permute(1, 2, 0).reshape(batch_size, n_vehicles, self.feature_size, len_pred).permute(0, 2, 1, 3)

        # fut = self._transformer_fut(fut, mask)
        attention_fut, pos_fut, distance_fut = self.fut_attention1.get_attention_matrices(pos, fut, mask)

        return attention_past, distance_past, attention_fut, pos_fut, distance_fut