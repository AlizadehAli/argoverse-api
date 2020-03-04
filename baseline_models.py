import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from conv1dRNN import Conv1dRNN
from LSTMpred import LSTMpred, LSTMpred2, GRUpred
import torch.jit as jit
from typing import List, Tuple
from torchdiffeq import odeint


embedding_size = 32
pred_hidden_size = 32
hist_length = 16
fut_length = 25
n_sample_per_time = 2
n_sample_max = 15


def outputActivation(x, mask, dim):
    # type: (Tensor, int) -> Tensor
    if x.shape[-1] > 5:
        p = x.narrow(dim, 5, 1)
        n_pred = p.shape[dim - 1]
        mask_veh = (1 - torch.prod(1 - mask, 0, keepdim=True)).view(1, x.shape[1], x.shape[2], 1, 1)
        p = p.masked_fill(mask_veh == 0, -1e9)
        p = nn.Softmax(dim-1)(p/(n_pred * n_pred))
    muX = x.narrow(dim, 0, 1)
    muY = x.narrow(dim, 1, 1)
    sigX = x.narrow(dim, 2, 1)
    sigY = x.narrow(dim, 3, 1)
    rho = x.narrow(dim, 4, 1)
    sigX = torch.exp(sigX/2)
    sigY = torch.exp(sigY/2)
    rho = torch.tanh(rho)

    out = torch.cat([muX, muY, sigX, sigY, rho, p], dim=dim)
    return out


class KalmanPredictor(nn.Module):

    def __init__(self, dt):
        super(KalmanPredictor, self).__init__()

        self.dt = dt
        self.n_var = 6

        self.angle_std = nn.Parameter(torch.ones(1) * np.pi/2)
        self.wheel_rate_noise = nn.Parameter(torch.ones(1) * np.pi/4)
        self.wheel_std = nn.Parameter(torch.ones(1) * np.pi/4)
        self.acceleration_noise = nn.Parameter(torch.ones(1) * 20)
        self.velocity_std = nn.Parameter(torch.ones(1) * 50)
        self.acceleration_std = nn.Parameter(torch.ones(1) * 50)

        self.LR_ = torch.eye(2, 2)+torch.randn(2, 2)*1.e-4
        self.LR_[0, 0] = 1.e-2
        self.LR_[1, 1] = 1.e-2
        self.LR_ = nn.Parameter(self.LR_)
        Q = torch.randn(self.n_var, self.n_var)*1.e-3
        Q[0, 0] = dt * dt / 4
        Q[1, 1] = dt * dt / 4
        Q[2, 2] = dt * dt * dt * dt * self.velocity_std * self.velocity_std / 4
        Q[3, 3] = dt / 3
        Q[4, 4] = dt * dt * dt * dt * self.velocity_std * self.velocity_std / 4
        Q[5, 5] = 1

        LQ = torch.cholesky(Q * self.acceleration_noise * self.acceleration_noise)
        self.LQ_ = nn.Parameter(LQ*dt)

        if torch.cuda.is_available():
            # state (x, y, theta, speed, acceleration, wheel_angle)
            self.H = torch.zeros(2, self.n_var).cuda() # observations are x, y
            self.B = torch.zeros(2).cuda() # actions are acceleration, wheel_angle variation
        else:
            self.H = torch.zeros(2, self.n_var)  # observations are x, y
            self.B = torch.zeros(2)  # actions are acceleration, wheel_angle variation

        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H_t = self.H.transpose(1, 0)
        self.B[0] = 1
        self.B[1] = 1

        self.debat = 2.6

    def _compute_hist_filter(self, Z):
        V0 = (Z[:, 1, :] - Z[:, 0, :]) / self.dt
        V1 = (Z[:, 2, :] - Z[:, 1, :]) / self.dt
        A0 = (V1 - V0) / self.dt
        X, P = self._kalman_init(Z[:, 0, :], V0, A0)
        for i in range(1, Z.shape[1]):
            X, P = self._kalman_pred(X, P)
            y, S = self._kalman_innovation(X, Z.narrow(1, i, 1).view(Z.shape[0], Z.shape[2]), P)
            X, P = self._kalman_update(X, P, S, y)

        return X, P

    def forward(self, hist, len_pred):
        X, P = self._compute_hist_filter(hist)
        if torch.cuda.is_available():
            pred_out = torch.zeros(X.shape[0], len_pred, 5).cuda()
        else:
            pred_out = torch.zeros(X.shape[0], len_pred, 5)
        for i in range(len_pred):
            X, P = self._kalman_pred(X, P)
            pred_out[:, i, :2] = torch.matmul(self.H, X.unsqueeze(2)).view(X.shape[0], self.H.shape[0])
            temp_out = torch.matmul(torch.matmul(self.H, P), self.H_t)
            pred_out[:, i, 2] = torch.sqrt(temp_out[:, 0, 0])
            pred_out[:, i, 3] = torch.sqrt(temp_out[:, 1, 1])
            pred_out[:, i, 4] = temp_out[:, 0, 1]/torch.sqrt(pred_out[:, i, 2].clone()*pred_out[:, i, 3].clone())
        return pred_out

    def _kalman_pred(self, X, P, u=None, Pu=None):
        if u is not None:
            assert Pu is not None # u and Pu are predicted actions and predicted noise over actions they should come together

        X_pred = X.clone()

        if u is not None:
            wheel_angles = X[:, 4] + self.B[0] * u[:, 0] / 2
            accelerations = X[:, 5] + self.B[1] * u[:, 1] / 2
        else:
            accelerations = X[:, 5]
            wheel_angles = X[:, 4]

        velocities = X[:, 3] + self.dt * accelerations / 2

        tan_w = torch.tan(wheel_angles)
        mask_big_angles = (torch.abs(tan_w) > 1.)

        curvature = tan_w / self.debat
        curvature[mask_big_angles] = 1 / torch.sqrt(self.debat * self.debat/(tan_w[mask_big_angles]**2) + 0.25*self.debat * self.debat)

        angles = X[:, 2] + self.dt*velocities*curvature / 2

        # y_pos = X[:, 1] + self.dt * velocities * torch.sin(angles) / 2
        # x_pos = X[:, 0] + self.dt * velocities * torch.cos(angles) / 2

        J = self._compute_jacobian(velocities, angles,
                                   curvature, tan_w, wheel_angles, mask_big_angles)

        x_pos = X[:, 0] + self.dt * X[:, 3] * torch.cos(X[:, 2])
        y_pos = X[:, 1] + self.dt * X[:, 3] * torch.sin(X[:, 2])

        angles = X[:, 2] + self.dt*X[:, 3]*curvature

        velocities = X[:, 3] + self.dt * X[:, 5]

        if u is not None:
            wheel_angles = X[:, 4] + self.dt * self.B[0] * u[:, 0]
            accelerations = X[:, 5] + self.dt * self.B[1] * u[:, 1]

        X_pred[:, 0] = x_pos
        X_pred[:, 1] = y_pos
        X_pred[:, 2] = angles
        X_pred[:, 3] = velocities
        X_pred[:, 4] = wheel_angles
        X_pred[:, 5] = accelerations

        P_pred = torch.matmul(torch.matmul(J, P), J.transpose(2, 1))

        if u is None:
            Q = torch.matmul(self.LQ_, self.LQ_.transpose(1, 0))
        else:
            Q = torch.matmul(self.LQ_, self.LQ_.transpose(1, 0))

        P_pred += Q
        return X_pred, P_pred

    def _compute_jacobian(self, velocities, angles, curvature, tan_w, wheel_angles, mask_big_angles):
        if torch.cuda.is_available():
            J = torch.eye(self.n_var).unsqueeze(0).repeat((velocities.shape[0], 1, 1)).cuda()
        else:
            J = torch.eye(self.n_var).unsqueeze(0).repeat((velocities.shape[0], 1, 1))

        J[:, 2, 0] = -self.dt*velocities*torch.sin(angles)
        J[:, 2, 1] = self.dt*velocities*torch.cos(angles)

        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos_w = torch.cos(wheel_angles)

        J[:, 3, 2] = self.dt*curvature
        J[:, 3, 1] = self.dt*(torch.sin(angles) + velocities*cos*J[:, 3, 2].clone())
        J[:, 3, 0] = self.dt*(torch.cos(angles) - velocities*sin*J[:, 3, 2].clone())

        # If big angles :
        tan_w_big = tan_w[mask_big_angles]
        cos_w_big = cos_w[mask_big_angles]
        J[mask_big_angles, 4, 2] = \
            self.dt * velocities[mask_big_angles] /\
            (cos_w_big*cos_w_big*2*torch.pow(1 + 0.25*tan_w_big*tan_w_big, 1.5)*self.debat)
        # Else if small angles :
        mask_small_angles = ~mask_big_angles
        J[mask_small_angles, 4, 2] = self.dt * velocities[mask_small_angles] / ((cos_w[mask_small_angles]**2) * self.debat)
        J[:, 4, 1] = self.dt * velocities * cos * J[:, 4, 2].clone()
        J[:, 4, 0] = - self.dt * velocities * sin * J[:, 4, 2].clone()

        J[:, 5, 3] = self.dt
        J[:, 5, 2] = self.dt*self.dt*curvature
        J[:, 5, 1] = self.dt * velocities * cos * J[:, 5, 2].clone()
        J[:, 5, 0] = -self.dt * velocities * sin * J[:, 5, 2].clone()

        # A[:, 0, 3] = torch.cos(angles + self.dt * yaw_rates / 2) * self.dt
        # A[:, 0, 4] = torch.cos(angles + self.dt * yaw_rates / 2) * self.dt ** 2 / 2
        # A[:, 1, 3] = torch.sin(angles + self.dt * yaw_rates / 2) * self.dt
        # A[:, 1, 4] = torch.sin(angles + self.dt * yaw_rates / 2) * self.dt ** 2 / 2
        # A[:, 3, 4] = self.dt
        #
        # JA = A
        # JA[:, 0, 2] = -torch.sin(angles + self.dt * yaw_rates / 2) * self.dt * (velocities + self.dt * accelerations / 2)
        # JA[:, 0, 5] = -self.dt ** 2 / 2 * torch.sin(angles + self.dt * yaw_rates / 2) * (velocities + self.dt * accelerations / 2) ** 2 * (
        #             1 + tan_w ** 2) / self.debat
        # JA[:, 1, 2] = torch.cos(angles + self.dt * yaw_rates) * self.dt * (velocities + self.dt * accelerations / 2)
        # JA[:, 1, 5] = self.dt ** 2 / 2 * torch.cos(angles + self.dt * yaw_rates / 2) * (velocities + self.dt * accelerations / 2) ** 2 * (
        #             1 + tan_w ** 2) / self.debat
        # JA[:, 2, 5] = (1 + tan_w ** 2) * (velocities + accelerations * self.dt / 2) / self.debat
        # JA[:, 2, 3] = self.dt * curvatures
        # JA[:, 2, 4] = self.dt ** 2 / 2 * curvatures
        #
        # corr = torch.zeros_like(JA)
        # corr[:, 0, 3] = -self.dt ** 2 / 2 * torch.sin(angles + self.dt * yaw_rates / 2) * self.dt * (velocities + self.dt * accelerations / 2) * curvatures
        # corr[:, 0, 4] = -self.dt ** 3 / 4 * torch.sin(angles + self.dt * yaw_rates / 2) * self.dt * (velocities + self.dt * accelerations / 2) * curvatures
        # corr[:, 1, 3] = self.dt ** 2 / 2 * torch.cos(angles + self.dt * yaw_rates / 2) * self.dt * (velocities + self.dt * accelerations / 2) * curvatures
        # corr[:, 1, 4] = self.dt ** 3 / 4 * torch.cos(angles + self.dt * yaw_rates / 2) * self.dt * (velocities + self.dt * accelerations / 2) * curvatures

        return J


    def _kalman_innovation(self, X, Z, P):
        R = torch.matmul(self.LR_, self.LR_.transpose(1, 0))
        y = Z - torch.matmul(self.H, X.unsqueeze(2)).view(X.shape[0], self.H.shape[0])
        S = torch.matmul(torch.matmul(self.H, P), self.H.transpose(1, 0)) + R
        return y, S

    def _kalman_update(self, X, P, S, y):
        K, _ = torch.solve(torch.matmul(self.H, P.transpose(2, 1)), S.transpose(2, 1))
        K = K.transpose(2, 1)
        X = X + torch.matmul(K, y.unsqueeze(2)).view(y.shape[0], K.shape[1])
        X[:, 5] = torch.fmod(X[:, 5], 2*np.pi)
        P = P - torch.matmul(torch.matmul(K, self.H), P)
        return X, P

    def _kalman_init(self, Z, V0, A0):
        angle = torch.atan2(V0[:, 1], V0[:, 0])
        # sign_acc = torch.sign(torch.sum(V0 * A0, 1))
        X = torch.matmul(self.H_t, Z.unsqueeze(2)).view(Z.shape[0], self.H_t.shape[0])
        X[:, 2] = angle
        X[:, 3] = torch.sqrt(torch.sum(V0*V0, 1))
        X[:, 4] = 0
        X[:, 5] = 0#sign_acc*torch.sqrt(torch.sum(A0*A0, 1))*0.5
        if torch.cuda.is_available():
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var).cuda()
        else:
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var)
        R = torch.matmul(self.LR_, self.LR_.transpose(1, 0))
        P += torch.chain_matmul(self.H_t, R, self.H)
        P[:, 2, 2] = self.angle_std*self.angle_std
        P[:, 3, 3] = self.velocity_std*self.velocity_std
        P[:, 4, 4] = self.wheel_std*self.wheel_std
        P[:, 5, 5] = self.acceleration_std*self.acceleration_std
        return X, P










class KalmanLSTMPredictor(nn.Module):

    def __init__(self, dt):
        super(KalmanLSTMPredictor, self).__init__()

        self.dt = dt
        self.n_var = 6
        self.offset_xy = 3
        self.n_layers = 3
        self.input_feature_size = self.n_var + 2 * self.offset_xy * self.offset_xy
        self.feature_size = 32

        self.max_accel_x = nn.Parameter(torch.ones(1) * 10)
        self.max_accel_y = nn.Parameter(torch.ones(1) * 20)
        self.velocity_std_x = nn.Parameter(torch.ones(1) * 10)
        self.velocity_std_y = nn.Parameter(torch.ones(1) * 50)
        self.acceleration_std_x = nn.Parameter(torch.ones(1) * 10)
        self.acceleration_std_y = nn.Parameter(torch.ones(1) * 30)

        self.R_x_ = torch.zeros(1)
        self.R_y_ = torch.zeros(1)
        self.R_x_[0] = 0.0448  # dt * self.velocity_std_x
        self.R_y_[0] = 1.e-3  # dt * self.velocity_std_y
        self.R_x_ = nn.Parameter(self.R_x_)
        self.R_y_ = nn.Parameter(self.R_y_)

        self.G_x_ = torch.randn(self.offset_xy)
        self.G_y_ = torch.randn(self.offset_xy)

        if torch.cuda.is_available():
            # state (x, y, theta, speed, acceleration, wheel_angle)
            self.H = torch.zeros(1, self.offset_xy).cuda()  # observations are x, y
            self.H_t = torch.zeros(self.offset_xy, 1).cuda()  # observations are x, y
            self.B = torch.zeros(1).cuda()  # actions are accelerations
            self.F = torch.eye(self.offset_xy).cuda()
        else:
            self.H = torch.zeros(1, self.offset_xy)  # observations are x, y
            self.H_t = torch.zeros(self.offset_xy, 1)  # observations are x, y
            self.B = torch.zeros(1)  # actions are accelerations
            self.F = torch.eye(self.offset_xy)

        self.H[0, 0] = 1
        # self.H[1, self.offset_xy] = 1
        self.H_t[0, 0] = 1
        # self.H_t[self.offset_xy, 1] = 1
        self.B[0] = dt

        self.F[0, 1] = dt
        self.F[0, 2] = dt * dt / 2
        self.F[1, 2] = dt

        self.G_x_[0] = 0.0059  # dt*dt*dt/4
        self.G_x_[1] = 0.2981  # dt*dt/3
        self.G_x_[2] = 0.0393  # dt
        self.G_y_ = self.G_x_.clone() / 5

        self.G_x_ = nn.Parameter(self.G_x_)
        self.G_y_ = nn.Parameter(self.G_y_)

        self.inputcell = nn.Linear(self.input_feature_size, self.feature_size)
        self.LSTMcells_enc = []
        for i in range(self.n_layers):
            self.LSTMcells_enc.append(nn.LSTMCell(self.feature_size,
                                                  self.feature_size))
        self.LSTMcells_enc = nn.ModuleList(self.LSTMcells_enc)

        self.LSTMcells_dec = []
        for i in range(self.n_layers):
            self.LSTMcells_dec.append(nn.LSTMCell(self.feature_size,
                                                  self.feature_size))
        self.LSTMcells_dec = nn.ModuleList(self.LSTMcells_dec)

        self.outputcell = nn.Linear(self.feature_size, 4)

    def set_kalman_trainable(self, is_trainable):
        self.max_accel_x.requires_grad = is_trainable
        self.max_accel_y.requires_grad = is_trainable
        self.velocity_std_x.requires_grad = is_trainable
        self.velocity_std_y.requires_grad = is_trainable
        self.acceleration_std_x.requires_grad = is_trainable
        self.acceleration_std_y.requires_grad = is_trainable
        self.R_x_.requires_grad = is_trainable
        self.R_y_.requires_grad = is_trainable
        self.G_x_.requires_grad = is_trainable
        self.G_y_.requires_grad = is_trainable

    def _compute_hist_filter(self, Z):
        batch_size = Z.shape[0]

        if torch.cuda.is_available():
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in
                       range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)).cuda() for i in
                       range(len(self.LSTMcells_dec))]
        else:
            hx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]
            cx_list = [Variable(torch.zeros(batch_size, self.feature_size)) for i in range(len(self.LSTMcells_dec))]

        Z_x = Z.narrow(2, 0, 1).view(batch_size, Z.shape[1])
        Z_y = Z.narrow(2, 1, 1).view(batch_size, Z.shape[1])

        X_x, P_x = self._kalman_init(Z_x[:, 0])
        X_y, P_y = self._kalman_init(Z_y[:, 0])
        for i in range(1, Z.shape[1]):
            XP = torch.cat([X_x, X_y,
                            P_x.view(batch_size, self.offset_xy * self.offset_xy),
                            P_y.view(batch_size, self.offset_xy * self.offset_xy)], 1)
            XP = torch.tanh(self.inputcell(XP))
            hx_list[0], cx_list[0] = self.LSTMcells_enc[0](XP, (hx_list[0], cx_list[0]))
            for j, cell in enumerate(self.LSTMcells_enc[1:]):
                hx_list[j + 1], cx_list[j + 1] = cell(cx_list[j], (hx_list[j + 1], cx_list[j + 1]))
            X_x, P_x = self._kalman_pred(X_x, P_x, self.max_accel_x, self.G_x_)
            X_y, P_y = self._kalman_pred(X_y, P_y, self.max_accel_y, self.G_y_)
            y_x, S_x = self._kalman_innovation(X_x, Z_x.narrow(1, i, 1).view(Z_x.shape[0]), P_x, self.R_x)
            y_y, S_y = self._kalman_innovation(X_y, Z_y.narrow(1, i, 1).view(Z_y.shape[0]), P_y, self.R_y)
            X_x, P_x = self._kalman_update(X_x, P_x, S_x, y_x)
            X_y, P_y = self._kalman_update(X_y, P_y, S_y, y_y)

        return (X_x, X_y), (P_x, P_y), (hx_list, cx_list)

    def forward(self, hist, len_pred):
        batch_size = hist.shape[0]
        self.R_x = torch.matmul(self.R_x_, self.R_x_)
        self.R_y = torch.matmul(self.R_y_, self.R_y_)

        (X_x, X_y), (P_x, P_y), (hx_list, cx_list) = self._compute_hist_filter(hist)

        pred_out = []
        for i in range(len_pred):

            XP = torch.cat([X_x, X_y,
                            P_x.view(batch_size, self.offset_xy * self.offset_xy),
                            P_y.view(batch_size, self.offset_xy * self.offset_xy)], 1)
            XP = torch.tanh(self.inputcell(XP))
            hx_list[0], cx_list[0] = self.LSTMcells_dec[0](XP, (hx_list[0], cx_list[0]))
            for j, cell in enumerate(self.LSTMcells_dec[1:]):
                hx_list[j + 1], cx_list[j + 1] = cell(cx_list[j], (hx_list[j + 1], cx_list[j + 1]))

            pred = self.outputcell(cx_list[-1])

            pred_x = pred.narrow(1, 0, 1).view(batch_size)
            pred_y = pred.narrow(1, 1, 1).view(batch_size)
            std_x = pred.narrow(1, 2, 1).view(batch_size)
            std_y = pred.narrow(1, 3, 1).view(batch_size)

            temp_out = []
            X_x, P_x = self._kalman_pred(X_x, P_x, self.max_accel_x, self.G_x_, pred_x, std_x)
            X_y, P_y = self._kalman_pred(X_y, P_y, self.max_accel_y, self.G_y_, pred_y, std_y)
            temp_out.append(torch.matmul(self.H, X_x.unsqueeze(2)))
            temp_out.append(torch.matmul(self.H, X_y.unsqueeze(2)))
            temp_out.append(torch.sqrt(torch.matmul(torch.matmul(self.H, P_x), self.H_t)))
            temp_out.append(torch.sqrt(torch.matmul(torch.matmul(self.H, P_y), self.H_t)))
            temp_out.append(temp_out[-1] * 0)
            pred_out.append(torch.cat(temp_out, 2))

        pred_out = torch.cat(pred_out, 1)
        return pred_out

    def _kalman_pred(self, X, P, max_accel, G, u=None, Su=None):
        assert (u is None) == (Su is None)

        if u is not None:
            X[:, 2] = self.B * u

        X_pred = torch.matmul(self.F, X.unsqueeze(2)).view(X.shape[0], self.F.shape[0])

        if u is None:
            Q = torch.matmul(G.unsqueeze(1) * max_accel, G.unsqueeze(0) * max_accel)
        else:
            Q = torch.matmul(G.unsqueeze(1) * Su.view(-1, 1, 1), G.unsqueeze(0) * Su.view(-1, 1, 1))

        P_pred = torch.matmul(torch.matmul(self.F, P), self.F.transpose(1, 0)) + Q

        return X_pred, P_pred

    def _kalman_innovation(self, X, Z, P, R):
        y = Z - torch.matmul(self.H, X.unsqueeze(2)).view(X.shape[0])
        S = torch.matmul(torch.matmul(self.H, P), self.H_t) + R
        return y, S

    def _kalman_update(self, X, P, S, y):
        K, _ = torch.solve(torch.matmul(self.H, P.transpose(2, 1)), S.transpose(2, 1))
        K = K.transpose(2, 1)
        X = X + (y.view(-1, 1, 1) * K).view(y.shape[0], K.shape[1])
        P = P - torch.matmul(torch.matmul(K, self.H), P)
        return X, P

    def _kalman_init(self, Z, is_x=True):
        X = (Z.view(-1, 1, 1) * self.H_t).view(Z.shape[0], self.H_t.shape[0])
        if torch.cuda.is_available():
            P = torch.zeros(Z.shape[0], self.offset_xy, self.offset_xy).cuda()
        else:
            P = torch.zeros(Z.shape[0], self.offset_xy, self.offset_xy)

        if is_x:
            P[:, 0, 0] = self.R_x
            P[:, 1, 1] = self.velocity_std_x * self.velocity_std_x
            P[:, 2, 2] = self.acceleration_std_x * self.acceleration_std_x
        else:
            P[:, 0, 0] = self.R_y
            P[:, 1, 1] = self.velocity_std_y * self.velocity_std_y
            P[:, 2, 2] = self.acceleration_std_y * self.acceleration_std_y

        return X, P



class KalmanCV(jit.ScriptModule):

    __constants__ = ['dt', 'n_var']

    def __init__(self, dt):
        super(KalmanCV, self).__init__()

        self.dt = dt
        self.n_var = 4

        self.velocity_std_x = nn.Parameter(torch.ones(1) * 2.58)
        self.velocity_std_y = nn.Parameter(torch.ones(1) * 20)
        self.acceleration_std_x = nn.Parameter(torch.ones(1) * 1.55)
        self.acceleration_std_y = nn.Parameter(torch.ones(1) * 5.76)

        self.GR_ = torch.randn(2) * 1e-3
        self.GR_ = nn.Parameter(self.GR_)

        if torch.cuda.is_available():
            self.H = jit.Attribute(torch.zeros(2, self.n_var, requires_grad=False).cuda(), torch.Tensor)  # observations are x, y
            self.H_t = jit.Attribute(torch.zeros(self.n_var, 2, requires_grad=False).cuda(), torch.Tensor)  # observations are x, y
            self.B = jit.Attribute(torch.zeros(self.n_var, 2, requires_grad=False).cuda(), torch.Tensor) # actions are accelerations
            self.F = jit.Attribute(torch.eye(self.n_var, requires_grad=False).cuda(), torch.Tensor)
            self.G_ = jit.Attribute(torch.zeros(self.n_var, requires_grad=False).cuda(), torch.Tensor)
            self.P0 = jit.Attribute(torch.zeros(51, self.n_var, self.n_var, requires_grad=False).cuda(), torch.Tensor)
            self.Id = jit.Attribute(torch.eye(self.n_var, requires_grad=False).cuda(), torch.Tensor)
            # self.H = torch.zeros(2, self.n_var, requires_grad=False).cuda()  # observations are x, y
            # self.H_t = torch.zeros(self.n_var, 2, requires_grad=False).cuda()  # observations are x, y
            # self.B = torch.zeros(self.n_var, 2, requires_grad=False).cuda() # actions are accelerations
            # self.F = torch.eye(self.n_var, requires_grad=False).cuda()
            # self.G_ = torch.zeros(self.n_var, requires_grad=False).cuda()
            # self.P0 = torch.zeros(1, self.n_var, self.n_var, requires_grad=False).cuda()

        else:
            self.H = jit.Attribute(torch.zeros(2, self.n_var, requires_grad=False), torch.Tensor)  # observations are x, y
            self.H_t = jit.Attribute(torch.zeros(self.n_var, 2, requires_grad=False), torch.Tensor)  # observations are x, y
            self.B = jit.Attribute(torch.zeros(self.n_var, 2, requires_grad=False), torch.Tensor) # actions are accelerations
            self.F = jit.Attribute(torch.eye(self.n_var, requires_grad=False), torch.Tensor)
            self.G_ = jit.Attribute(torch.zeros(self.n_var, requires_grad=False), torch.Tensor)
            self.P0 = jit.Attribute(torch.zeros(51, self.n_var, self.n_var, requires_grad=False), torch.Tensor)
            self.Id = jit.Attribute(torch.eye(self.n_var, requires_grad=False), torch.Tensor)
            # self.H = torch.zeros(2, self.n_var, requires_grad=False)  # observations are x, y
            # self.H_t = torch.zeros(self.n_var, 2, requires_grad=False)  # observations are x, y
            # self.B = torch.zeros(self.n_var, 2, requires_grad=False)  # actions are accelerations
            # self.F = torch.eye(self.n_var, requires_grad=False)
            # self.G_ = torch.zeros(self.n_var, requires_grad=False)
            # self.P0 = torch.zeros(1, self.n_var, self.n_var, requires_grad=False)

        self.G_[0] = dt * dt / 2
        self.G_[1] = dt
        self.G_[2] = dt * dt / 2
        self.G_[3] = dt
        coef_G = torch.zeros(self.n_var)
        coef_G[0] = -1.7168
        coef_G[1] = 0.1378
        coef_G[2] = -0.0294
        coef_G[3] = -0.4126
        self.coef_G = nn.Parameter(coef_G)

        # Observation matrix that mask speeds and keeps positions
        self.H[0, 0] = 1
        self.H[1, 2] = 1
        self.H_t[0, 0] = 1
        self.H_t[2, 1] = 1

        # Transition matrix that defines evolution of position over a time step for a given state
        self.F[0, 1] = dt
        self.F[2, 3] = dt

    # @jit.script_method
    def _compute_hist_filter(self, Z):
##        # type: (Tensor) -> Tuple[Tensor, Tensor]
        V0 = (Z[1] - Z[0])/self.dt
        X, P = self._kalman_init(Z[0], V0)
        for i in range(1, len(Z)):
            X, P = self._kalman_pred(X, P)
            y, S = self._kalman_innovation(X, Z[i], P)
            X, P = self._kalman_update(X, P, S, y)
        return X, P

    # @jit.script_method
    def forward(self, hist, len_pred):
##        # type: (Tensor, int) -> Tensor
        batch_size = hist.shape[1]
        hist = hist.unsqueeze(3)
        X, P = self._compute_hist_filter(hist)

        pred_mu = torch.jit.annotate(List[torch.Tensor], [])
        pred_P = torch.jit.annotate(List[torch.Tensor], [])
        for i in range(len_pred):
            X, P = self._kalman_pred(X, P)
            temp_X_out = torch.matmul(self.H, X).transpose(2, 1)
            temp_P_out = torch.matmul(torch.matmul(self.H, P), self.H_t)
            pred_mu += [temp_X_out]
            pred_P += [temp_P_out]

        pred_mu = torch.stack(pred_mu)
        pred_P = torch.stack(pred_P)
        pred_P = pred_P.view(len_pred, batch_size, 2, 2)
        pred_mu = pred_mu.view(len_pred, batch_size, 2)
        sigma_x = torch.sqrt(pred_P[:, :, 0, 0].view(len_pred, batch_size, 1))
        sigma_y = torch.sqrt(pred_P[:, :, 1, 1].view(len_pred, batch_size, 1))
        rho = (pred_P[:, :, 0, 1] + pred_P[:, :, 1, 0]).view(len_pred, batch_size, 1)/(2*sigma_x*sigma_y)
        return torch.cat([pred_mu, sigma_x, sigma_y, rho], 2)

    @jit.script_method
    def _kalman_pred(self, X, P):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        X_pred = torch.matmul(self.F, X)

        Q = torch.matmul((self.G_*self.coef_G).unsqueeze(1), (self.G_*self.coef_G).unsqueeze(0))
        Q[:2, :2] *= self.acceleration_std_x * self.acceleration_std_x
        Q[2:, 2:] *= self.acceleration_std_y * self.acceleration_std_y

        P_pred = torch.matmul(torch.matmul(self.F, P), self.F.transpose(1, 0)) + Q

        return X_pred, P_pred

    @jit.script_method
    def _kalman_innovation(self, X, Z, P):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        R = torch.matmul(self.GR_.unsqueeze(1), self.GR_.unsqueeze(0))
        y = Z - torch.matmul(self.H, X)
        S = torch.matmul(torch.matmul(self.H, P), self.H_t) + R
        return y, S

    @jit.script_method
    def _kalman_update(self, X, P, S, y):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

        # S_inv = torch.inverse(S)
        # K = torch.matmul(torch.matmul(P, self.H_t), S_inv)
        R = torch.matmul(self.GR_.unsqueeze(1), self.GR_.unsqueeze(0))
        K, _ = torch.solve(torch.matmul(self.H, P.transpose(2, 1)), S.transpose(2, 1))
        K = K.transpose(2, 1)
        X = X + torch.matmul(K, y)

        # Joseph formula for stability
        ImKH = self.Id.unsqueeze(0) - torch.matmul(K, self.H)
        KRK = torch.matmul(torch.matmul(K, R), K.transpose(2, 1))
        P = torch.matmul(torch.matmul(ImKH, P), ImKH.transpose(2, 1)) + KRK
        return X, P

    # @jit.script_method
    def _kalman_init(self, Z, V):
##        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        X = torch.matmul(self.H_t, Z)
        X[:, 1] = V[:, 0]
        X[:, 3] = V[:, 1]

        if torch.cuda.is_available():
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var).cuda()
        else:
            P = torch.zeros(Z.shape[0], self.n_var, self.n_var)
        # if self.P0.shape[0] != Z.shape[0]:
        #     self.P0 = self.P0[0, :, :].repeat((Z.shape[0], 1, 1))
        # P = self.P0.clone()
        R = torch.matmul(self.GR_.unsqueeze(1), self.GR_.unsqueeze(0))

        P[:, 0, 0] = R[0, 0]
        P[:, 1, 1] = self.velocity_std_x * self.velocity_std_x
        P[:, 2, 2] = R[1, 1]
        P[:, 3, 3] = self.velocity_std_y * self.velocity_std_y

        return X, P























