from data_loader import ArgoDataset
from loss_functions import multiMSE, multiNLL, simpleNLL, simpleMSE
from baseline_models import KalmanLSTMPredictor, KalmanPredictor, KalmanCV
from torch.utils.data import DataLoader
import torch

root_dir = 'train/data/'
batch_size = 4
dt=0.1
n_epoch = 1
lr = 0.0001

train_set = ArgoDataset(root_dir)

train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=0, collate_fn=train_set.collate_fn)
net = KalmanCV(dt)
if torch.cuda.is_available():
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(n_epoch):
    it_train = iter(train_data_loader)

    len_train = len(it_train)
    avg_loss = 0
    avg_mse = 0
    for i in range(len_train):
        x, y, mask_x, mask_y = next(it_train)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
            mask_x = mask_x.cuda()
            mask_y = mask_y.cuda()

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3])
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2], y.shape[3])
        mask_y = mask_y.reshape(mask_y.shape[0], -1)
        y_pred = net(x, 30)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = simpleNLL(y_pred, y, mask_y, dim=2)
        mse = simpleMSE(y_pred, y, mask_y, dim=2)
        avg_loss += loss.detach()
        avg_mse += mse.detach()

        if i%100 == 99:
            torch.save(net.state_dict(), 'test_baseline.tar')
            avg_loss = avg_loss.item()
            print("Epoch no:", epoch, "| progress(%):",
                format(i / len_train / batch_size * 100, '0.2f'),
                '| loss:', format(avg_loss / 100, '0.4f'),
                '| mse:', format(avg_mse / 100, '0.4f'))
            avg_loss = 0
            avg_mse = 0
