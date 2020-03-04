import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Cellpred(nn.Module):

    def __init__(self, cell, feature_size, time_span, n_layers=3):
        super(Cellpred, self).__init__()
        self.diff_mean_value = 1.0
        self.time_span = time_span
        self.feature_size = feature_size
        self.Cells = []
        for i in range(n_layers):
            self.Cells.append(cell(feature_size, feature_size))
            self.init_weights(self.Cells[-1])
        self.Cells = nn.ModuleList(self.Cells)
        self.init_weights(self.Cells[-1])

    def init_weights(self, lay):
        for name, param in lay.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param, 0.9)

    def forward(self, seq_input, fut_len):
        batch_size = seq_input.shape[0]
        if torch.cuda.is_available():
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len).cuda()
        else:
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len)

        hist_len = seq_input.shape[2]
        if torch.cuda.is_available():
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.Cells))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.Cells))]
        else:
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.Cells))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.Cells))]
        for i in range(hist_len):
            hx_list[0], cx_list[0] = self.Cells[0](
                seq_input.narrow(2, i, 1).view(seq_input.shape[0], seq_input.shape[1]), (hx_list[0], cx_list[0]))
            for j, rnn in enumerate(self.Cells[1:]):
                hx_list[j+1], cx_list[j+1] = rnn(hx_list[j], (hx_list[j+1], cx_list[j+1]))
        recurrent_input = cx_list[-1]#*self.diff_mean_value + seq_input.narrow(2, hist_len-1, 1).squeeze()
        for i in range(fut_len):
            hx_list[0], cx_list[0] = self.Cells[0](recurrent_input, (hx_list[0], cx_list[0]))
            for j, rnn in enumerate(self.Cells[1:]):
                hx_list[j + 1], cx_list[j + 1] = rnn(hx_list[j], (hx_list[j + 1], cx_list[j + 1]))
            # recurrent_input = recurrent_input + cx_list[-1]*self.diff_mean_value
            recurrent_input = hx_list[-1]
            sub_sequence_output = sequence_output.narrow(2, i, 1).view(sequence_output.shape[0],
                                                                       sequence_output.shape[1])
            sub_sequence_output += recurrent_input
        pred_output = sequence_output
        return pred_output


class LSTMpred(Cellpred):

    def __init__(self, feature_size, time_span, n_layers=3):
        super(LSTMpred, self).__init__(nn.LSTMCell, feature_size, time_span, n_layers)


class GRUpred(Cellpred):

    def __init__(self, feature_size, time_span, n_layers=3):
        super(GRUpred, self).__init__(nn.GRUCell, feature_size, time_span, n_layers)

    def forward(self, seq_input, fut_len):
        batch_size = seq_input.shape[0]
        if torch.cuda.is_available():
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len).cuda()
        else:
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len)
        hist_len = seq_input.shape[2]
        if torch.cuda.is_available():
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.Cells))]
        else:
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.Cells))]
        for i in range(hist_len):
            hx_list[0] = self.Cells[0](seq_input.narrow(2, i, 1).view(seq_input.shape[0], seq_input.shape[1]),
                                       hx_list[0])
            for j, rnn in enumerate(self.Cells[1:]):
                hx_list[j+1] = rnn(hx_list[j], hx_list[j+1])
        recurrent_input = hx_list[-1]#*self.diff_mean_value + seq_input.narrow(2, hist_len-1, 1).squeeze()
        for i in range(fut_len):
            hx_list[0] = self.Cells[0](recurrent_input, hx_list[0])
            for j, rnn in enumerate(self.Cells[1:]):
                hx_list[j + 1] = rnn(hx_list[j], hx_list[j + 1])
            # recurrent_input = recurrent_input + hx_list[-1]*self.diff_mean_value
            recurrent_input = hx_list[-1]
            sub_sequence_output = sequence_output.narrow(2, i, 1).view(sequence_output.shape[0],
                                                                       sequence_output.shape[1])
            sub_sequence_output += recurrent_input
        pred_output = sequence_output
        return pred_output


class LSTMpred2(nn.Module):

    def __init__(self, feature_size, time_span, n_layers=3):
        super(LSTMpred2, self).__init__()
        self.diff_mean_value = 1.0
        self.time_span = time_span
        self.feature_size = feature_size
        # self.LSTMcells_enc = []
        # for i in range(n_layers):
        #     self.LSTMcells_enc.append(nn.LSTMCell(feature_size, feature_size))
        # self.LSTMcells_enc = nn.ModuleList(self.LSTMcells_enc)
        self.LSTMenc = nn.LSTM(feature_size, feature_size, n_layers)
        self.LSTMcells_dec = []
        for i in range(n_layers):
            self.LSTMcells_dec.append(nn.LSTMCell(feature_size, feature_size))
        self.LSTMcells_dec = nn.ModuleList(self.LSTMcells_dec)
        self.output_layer = nn.Conv1d(feature_size, feature_size, kernel_size=1)

    def forward(self, seq_input, fut_len):
        batch_size = seq_input.shape[0]
        if torch.cuda.is_available():
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len).cuda()
        else:
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len)

        hist_len = seq_input.shape[2]
        if torch.cuda.is_available():
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.LSTMcells_dec))]
        else:
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]

        # for i in range(hist_len):
        #     hx_list[0], cx_list[0] = self.LSTMcells_enc[0](
        #         seq_input.narrow(2, i, 1).view(seq_input.shape[0], seq_input.shape[1]), (hx_list[0], cx_list[0]))
        #     for j, rnn in enumerate(self.LSTMcells_enc[1:]):
        #         hx_list[j+1], cx_list[j+1] = rnn(cx_list[j], (hx_list[j+1], cx_list[j+1]))
        seq_input = seq_input.permute(2, 0, 1)
        _, (hx, cx) = self.LSTMenc(seq_input.narrow(0, 0, seq_input.shape[0]-1))
        for i in range(len(self.LSTMcells_dec)):
            hx_list[i] = hx[i]
            cx_list[i] = cx[i]
        recurrent_input = seq_input.narrow(0, seq_input.shape[0]-2, 1).view(seq_input.shape[1], seq_input.shape[2])
        for i in range(fut_len):
            hx_list[0], cx_list[0] = self.LSTMcells_dec[0](recurrent_input, (hx_list[0], cx_list[0]))
            for j, rnn in enumerate(self.LSTMcells_dec[1:]):
                hx_list[j + 1], cx_list[j + 1] = rnn(hx_list[j], (hx_list[j + 1], cx_list[j + 1]))
            recurrent_input = hx_list[-1]
            sub_sequence_output = sequence_output.narrow(2, i, 1).view(sequence_output.shape[0],
                                                                       sequence_output.shape[1])
            sub_sequence_output += recurrent_input
        pred_output = self.output_layer(sequence_output)
        return pred_output


class LSTMpred2_res(nn.Module):

    def __init__(self, feature_size, time_span, n_layers=3):
        super(LSTMpred2_res, self).__init__()
        self.diff_mean_value = 1.0
        self.time_span = time_span
        self.feature_size = feature_size
        # self.LSTMcells_enc = []
        # for i in range(n_layers):
        #     self.LSTMcells_enc.append(nn.LSTMCell(feature_size, feature_size))
        # self.LSTMcells_enc = nn.ModuleList(self.LSTMcells_enc)
        self.LSTMenc = nn.LSTM(feature_size, feature_size, n_layers)
        self.LSTMcells_dec = []
        for i in range(n_layers):
            self.LSTMcells_dec.append(nn.LSTMCell(feature_size, feature_size))
        self.LSTMcells_dec = nn.ModuleList(self.LSTMcells_dec)
        self.output_layer = nn.Conv1d(feature_size, feature_size, kernel_size=1)

    def forward(self, seq_input, fut_len):
        batch_size = seq_input.shape[0]
        if torch.cuda.is_available():
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len).cuda()
        else:
            sequence_output = torch.zeros(seq_input.shape[0], self.feature_size, fut_len)

        hist_len = seq_input.shape[2]
        if torch.cuda.is_available():
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in range(len(self.LSTMcells_dec))]
        else:
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]

        # for i in range(hist_len):
        #     hx_list[0], cx_list[0] = self.LSTMcells_enc[0](
        #         seq_input.narrow(2, i, 1).view(seq_input.shape[0], seq_input.shape[1]), (hx_list[0], cx_list[0]))
        #     for j, rnn in enumerate(self.LSTMcells_enc[1:]):
        #         hx_list[j+1], cx_list[j+1] = rnn(cx_list[j], (hx_list[j+1], cx_list[j+1]))
        seq_input = seq_input.permute(2, 0, 1)
        _, (hx, cx) = self.LSTMenc(seq_input.narrow(0, 0, hist_len-1))
        for i in range(len(self.LSTMcells_dec)):
            hx_list[i] = hx[i]
            cx_list[i] = cx[i]
        recurrent_input = seq_input.narrow(0, hist_len-2, 1).view(seq_input.shape[1], seq_input.shape[2])
        for i in range(fut_len):
            hx_list[0], cx_list[0] = self.LSTMcells_dec[0](recurrent_input, (hx_list[0], cx_list[0]))
            for j, rnn in enumerate(self.LSTMcells_dec[1:]):
                hx_list[j + 1], cx_list[j + 1] = rnn(hx_list[j], (hx_list[j + 1], cx_list[j + 1]))
            recurrent_input = recurrent_input.clone() + hx_list[-1]
            sub_sequence_output = sequence_output.narrow(2, i, 1).view(sequence_output.shape[0],
                                                                       sequence_output.shape[1])
            sub_sequence_output += recurrent_input
        pred_output = self.output_layer(sequence_output)
        return pred_output


