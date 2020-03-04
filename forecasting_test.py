from bicycle_predictor import Bicycle_model
import torch

dt = 0.1
batch_size = 512

# model = jit.script(bicycle_model(batch_size=batch_size, dt=dt))
model = Bicycle_model(batch_size=batch_size, dt=dt)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print('Using device ' + torch.cuda.get_device_name())
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
# optimizer = Ranger(model.parameters(), lr=0.0003)


def simpleNLL(y_pred, y_gt, mask=None, dim=2):
    eps = 1e-1
    eps_rho = 1e-2
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    sigX = torch.relu(y_pred.narrow(dim, 2, 1) - eps) + eps
    sigY = torch.relu(y_pred.narrow(dim, 3, 1) - eps) + eps
    rho = y_pred.narrow(dim, 4, 1)
    ohr = 1 / (torch.relu(1 - rho * rho - eps_rho) + eps_rho)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    nll = 0.5 * ohr * (torch.pow(diff_x / sigX, 2) + torch.pow(diff_y / sigY, 2) -
                       2 * rho * (diff_x / sigX) * (diff_y / sigY)) + \
          torch.log(sigX * sigY) - 0.5 * torch.log(ohr)
    if mask is None:
        lossVal = torch.mean(nll)
    else:
        nll = nll * mask.unsqueeze(dim).float()
        lossVal = torch.sum(nll) / torch.sum(mask)
    return lossVal


def prediction(inputs, num_points=30, avg_points=1):
    inputs = torch.from_numpy(inputs.astype('float32')).to(device)
    results = model(inputs, num_points)
    return results.cpu().numpy()


def step_train(inputs, num_hist=20):
    inputs = torch.from_numpy(inputs.astype('float32')).to(device)
    truth = inputs[:, num_hist:]
    inputs = inputs[:, :num_hist]
    num_points = truth.shape[1]
    pred = model(inputs, num_points)
    loss = simpleNLL(pred[:, num_hist:], truth)
    loss.requires_grad = True
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
    optimizer.step()


def xy_to_input(trajectory):
    traj_diff = trajectory[1:] - trajectory[:-1]
    yaw = np.arctan2(traj_diff[:, 1:2], traj_diff[:, 0:1])
    yaw = np.concatenate((yaw[0:1, :], yaw), axis=0)
    inputs = np.concatenate((trajectory, yaw), axis=1)
    return inputs

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader

##set root_dir to the correct path to your dataset folder
# root_dir = '../test_obs/data/'
# root_dir = '../forecasting_sample/data/'
root_dir = './train/data/'

afl = ArgoverseForecastingLoader(root_dir)

print('Total number of sequences:', len(afl))

import numpy as np


def get_multiple_forecasts(afl):
    predict_traj = []
    for i in range(9, 3, -1):
        predict_traj.append(prediction(afl.agent_traj, avg_points=(i)))
    return np.array(predict_traj)

## Compute all the results

# ax = plt.gca()

# output_all = {}
inputs = []
for epoch in range(10):
    for i, data in enumerate(afl):
        print('\r' + 'epoch: ' + str(epoch) + ' | iter: ' + str(i) + '/' + str(len(afl)), end="")
        inputs.append(xy_to_input(data.agent_traj))
        if i % batch_size == batch_size-1:
            # predict_traj = prediction(np.stack(inputs)[:, :20])
            step_train(np.stack(inputs))
            # for j in range(batch_size):
            #     input_traj = inputs[j]
            #
            #     seq_id = int(data.current_seq.name[:-4])
            #     counter += 1
                # plt.plot(input_traj[0, 0], input_traj[0, 1], '-o', c='r')  # starting point here
                # plt.plot(input_traj[:, 0], input_traj[:, 1], '+', c='b')
                # plt.plot(predict_traj[j, :20, 0], predict_traj[j, :20, 1], '-', c='b')
                #
                # plt.plot(predict_traj[j, 20:, 0], predict_traj[j, 20:, 1], '-', c='r')
                # plt.plot(predict_traj[j, 20:, 0], predict_traj[j, 20:, 1], '+', c='r')
                #
                # plt.xlabel('map_x_coord (m)')
                # plt.ylabel('map_y_coord (m)')
                # ax.set_aspect('equal')
                # plt.show()
            inputs = []
    torch.save(model.state_dict(), 'bicycle.tar')



