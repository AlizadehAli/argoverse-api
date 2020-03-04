import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1dRNN(nn.Module):
    def __init__(self, feature_size, time_span, n_layers=3, activation=nn.LeakyReLU(0.1)):
        super(Conv1dRNN, self).__init__()
        self.diff_mean_value = 0.01
        self.time_span = time_span
        self.activation = activation
        self.feature_size = feature_size
        self.lay = []
        for i in range(n_layers-1):
            self.lay.append(nn.Linear(feature_size * self.time_span, feature_size * self.time_span))
        self.lay.append(nn.Linear(feature_size * self.time_span, feature_size))
        self.lay = nn.ModuleList(self.lay)

    def forward(self, seq_input, n_steps):
        sequence_output = torch.zeros(seq_input.shape[0], self.feature_size,
                                      self.time_span + n_steps).cuda()

        sub_sequence_output = sequence_output.narrow(2, 0, self.time_span)
        sub_sequence_output += seq_input

        start = sequence_output.shape[2] - n_steps
        for i in range(n_steps):
            next_elem = self.step(sequence_output, i)
            sub_sequence = sequence_output.narrow(2, i + self.time_span, 1).squeeze()
            sub_sequence += next_elem

        pred_output = sequence_output.narrow(2, start, n_steps)

        return pred_output

    def step(self, sequence_input, i):
        part_to_use = sequence_input.narrow(2, i, self.time_span)
        enc = part_to_use.contiguous().view(part_to_use.shape[0], -1)
        for i in range(len(self.lay) - 1):
            enc = self.activation(self.lay[i](enc))
        enc = self.lay[len(self.lay)-1](enc)
        return enc*self.diff_mean_value + sequence_input.narrow(2, i+self.time_span-1, 1).squeeze()
