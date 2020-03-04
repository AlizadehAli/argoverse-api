from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
# import torch.nn.functional as F
import h5py
import torch.jit as jit
import torch.nn as nn



## Helper function for log sum exp calculation:
@jit.script
def logsumexp(inputs, mask, dim, keepdim=False):
    # type: (Tensor, Tensor, int, bool) -> Tensor
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    if mask is None:
        outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    else:
        # mask_veh = mask.unsqueeze(3)
        inputs_s = (inputs - s)#.masked_fill(mask_veh == 0, -1e3)
        outputs = s + inputs_s.exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


## Custom activation for output layer
@jit.script
def outputActivation(x, dim=2):
    # type: (Tensor, int) -> Tensor
    muX = x.narrow(dim, 0, 1)
    muY = x.narrow(dim, 1, 1)
    sigX = x.narrow(dim, 2, 1)
    sigY = x.narrow(dim, 3, 1)
    rho = x.narrow(dim, 4, 1)
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=dim)
    return out


### Dataset class for the NGSIM dataset
class SimpleNGSIMDataset(Dataset):

    def __init__(self, mat_traj_file, mat_tracks_file, t_h=30, t_f=50, d_s=2, normalize=True):
        self.D = np.array(h5py.File(mat_traj_file)['traj']).transpose()
        self.T = scp.loadmat(mat_tracks_file)['tracks']
        if normalize:
            self.mean, self.std = self.normalize_mean_std()
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array(self.mean).astype('float32')).cuda()
                self.std = torch.from_numpy(np.array(self.std).astype('float32')).cuda()
            else:
                self.mean = torch.from_numpy(np.array(self.mean).astype('float32'))
                self.std = torch.from_numpy(np.array(self.std).astype('float32'))
        else:
            self.mean = 0
            self.std = 1
        self.t_h = t_h  # length of track history
        self.t_f = t_f  # length of predicted trajectory
        self.d_s = d_s  # down sampling rate of all sequences

    def normalize_mean_std(self):
        mean = np.array([[0], [0]])
        std = np.array([[1], [1]])

        for i in range(len(self.T)):
            for j in range(len(self.T[i])):
                if self.T[i][j].size > 0:
                    self.T[i][j][1:3, :] = (self.T[i][j][1:3, :] - mean)/std
        return mean, std

    ## NLL for sequence, outputs sequence of NLL values for each time-step, uses mask for variable output lengths, used for evaluation
    def simpleNLLTest(self, fut_pred, proba_man, fut, mask=None, avg_along_time=True):
        if torch.cuda.is_available():
            acc = torch.zeros(len(proba_man), fut_pred[0].shape[0], fut_pred[0].shape[1]).cuda()
        else:
            acc = torch.zeros(len(proba_man), fut_pred[0].shape[0], fut_pred[0].shape[1])
        eps = 1e-3
        sum = 0
        for k in range(len(proba_man)):
            sum += proba_man[k]

        for k in range(len(proba_man)):
            wts = proba_man[k]
            y_pred = fut_pred[k]
            y_pred_pos = fut_pred[k].narrow(2, 0, 2)
            y_gt = fut
            # y_pred_pos = torch.cumsum(y_pred_pos, 0)# / y_pred_pos.shape[0]
            # y_gt = torch.cumsum(fut, 0)# / y_pred.shape[0]
            muX = y_pred_pos.narrow(2, 0, 1)
            muY = y_pred_pos.narrow(2, 1, 1)
            sigX = torch.relu(y_pred.narrow(2, 2, 1) - eps) + eps
            sigY = torch.relu(y_pred.narrow(2, 3, 1) - eps) + eps
            rho = y_pred.narrow(2, 4, 1)
            ohr = 1 / (torch.relu(1 - rho * rho - eps) + eps)
            x = y_gt.narrow(2, 0, 1)
            y = y_gt.narrow(2, 1, 1)
            diff_x = (x - muX) * self.std[0, 0]
            diff_y = (y - muY) * self.std[1, 0]
            sigX = sigX * self.std[0, 0]
            sigY = sigY * self.std[1, 0]
            out = -ohr * (diff_x * diff_x / (sigX * sigX) + diff_y * diff_y / (sigY * sigY) -
                          2 * rho * diff_x * diff_y / (sigX * sigY)) - \
                  2*torch.log(sigX * sigY) + torch.log(ohr) - torch.log(2*np.pi)
            acc[k, :, :] = 0.5 * out.squeeze() + torch.log(torch.relu(wts - 1e-6) + 1e-6) + np.log(2*np.pi)

        acc = -logsumexp(acc, mask, dim=0)
        # acc = acc
        if avg_along_time:
            lossVal = torch.mean(acc)
            return lossVal
        else:
            raise RuntimeError


    def unnormed_NLL(self, y_pred, y_gt, mask=None):
        eps = 1e-6
        eps_rho = 1e-2
        y_pred_pos = y_pred.narrow(2, 0, 2)
        # y_pred_pos = torch.cumsum(y_pred_pos, 0)
        # y_gt = torch.cumsum(y_gt, 0)
        muX = y_pred_pos.narrow(2, 0, 1)
        muY = y_pred_pos.narrow(2, 1, 1)
        sigX = torch.relu(y_pred.narrow(2, 2, 1) - eps) + eps
        sigY = torch.relu(y_pred.narrow(2, 3, 1) - eps) + eps
        rho = y_pred.narrow(2, 4, 1)
        ohr = 1 / (torch.relu(1 - rho * rho - eps_rho) + eps_rho)
        x = y_gt.narrow(2, 0, 1)
        y = y_gt.narrow(2, 1, 1)
        diff_x = (x - muX)*self.std[0, 0]
        diff_y = (y - muY)*self.std[1, 0]
        sigX *= self.std[0, 0]
        sigY *= self.std[1, 0]
        out = ohr * (diff_x * diff_x / (sigX * sigX) + diff_y * diff_y / (sigY * sigY) -
                     2 * rho * diff_x * diff_y / (sigX * sigY)) + 2*torch.log(sigX * sigY) - torch.log(ohr)
        acc = 0.5 * out
        if mask is None:
            lossVal = torch.mean(acc)
        else:
            lossVal = torch.sum(acc*mask.unsqueeze(2))/torch.sum(mask)
        return lossVal

    def unnormed_MSE(self, y_pred, y_gt, mask):
        y_pred_pos = y_pred.narrow(2, 0, 2)
        # y_pred_pos = torch.cumsum(y_pred_pos, 0)
        # y_gt = torch.cumsum(y_gt, 0)
        muX = y_pred_pos.narrow(2, 0, 1)
        muY = y_pred_pos.narrow(2, 1, 1)
        x = y_gt.narrow(2, 0, 1)
        y = y_gt.narrow(2, 1, 1)
        diff_x = (x - muX) * self.std[0, 0]
        diff_y = (y - muY) * self.std[1, 0]
        if mask is None:
            output = torch.mean(diff_x * diff_x + diff_y * diff_y)
        else:
            output = torch.sum((diff_x * diff_x + diff_y * diff_y)*mask)/torch.sum(mask)
        return output

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]

        hist = self.getHistory(vehId, t, vehId, dsId)
        fut = self.getFuture(vehId, t, dsId)

        return hist, fut


    ## Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    ## Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    ## Collate function for dataloader
    def collate_fn(self, samples):

        maxlen = self.t_h // self.d_s + 1

        # Initialize history, future
        time_size = self.t_f // self.d_s
        hist_batch = torch.zeros(maxlen, len(samples), 2)
        fut_batch = torch.zeros(time_size, len(samples), 2)

        for sampleId, (hist, fut) in enumerate(samples):

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
            hist_batch[0:len(hist), sampleId, 0] = torch.from_numpy(hist[:, 0])
            hist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1])
            fut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            fut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])

        return hist_batch, fut_batch

@jit.script
def simpleNLL(y_pred, y_gt, mask, dim=3):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    eps = 1e-1
    eps_rho = 1e-2
    y_pred_pos = y_pred.narrow(dim, 0, 2)
    muX = y_pred_pos.narrow(dim, 0, 1)
    muY = y_pred_pos.narrow(dim, 1, 1)
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
        nll = nll.masked_fill(mask.unsqueeze(dim) == 0, 0)
        lossVal = torch.sum(nll) / torch.sum(mask)
    return lossVal


def multiDPP(y_pred, mask):
    eps = 1e-3
    k = 1
    y = y_pred.narrow(4, 0, 2)
    p = torch.relu(y_pred.narrow(4, 5, 1) - eps) + eps

    distance_matrix = y.unsqueeze(4).repeat((1, 1, 1, 1, y.shape[3], 1))
    distance_matrix = distance_matrix - distance_matrix.transpose(4, 3)
    distance_matrix = p*torch.sum(distance_matrix * distance_matrix, 5)
    # No quality estimation... replaced with p
    L = torch.exp(-k * torch.matmul(distance_matrix, distance_matrix.transpose(4, 3)))
    n_pred = L.shape[-1]
    if torch.cuda.is_available():
        I = torch.eye(n_pred).cuda()
    else:
        I = torch.eye(n_pred)
    I = I.view(1, 1, 1, n_pred, n_pred)
    loss_mat = torch.sum(torch.sum((I - torch.inverse(I + L)) * I, dim=4), dim=3)
    mask = mask.masked_fill(torch.isnan(loss_mat), 0)
    loss_mat = loss_mat.masked_fill(mask == 0, 0)
    loss = -torch.sum(loss_mat) / torch.sum(mask)
    return loss

#
# def multiDPP(y_pred, mask):
#     eps = 1e-3
#     k = 1e-4
#     y = y_pred.narrow(4, 0, 2)
#     p = torch.relu(y_pred.narrow(4, 5, 1) - eps) + eps
#
#     distance_matrix = y.unsqueeze(4).repeat((1, 1, 1, 1, y.shape[3], 1))
#     distance_matrix = distance_matrix - distance_matrix.transpose(4, 3)
#     distance_matrix = p * torch.sum(distance_matrix * distance_matrix, 5)
#     # No quality estimation... replaced with p
#     L = torch.exp(-k * distance_matrix)
#     n_pred = L.shape[-1]
#     if torch.cuda.is_available():
#         I = torch.eye(n_pred).cuda()
#     else:
#         I = torch.eye(n_pred)
#     I = I.view(1, 1, 1, n_pred, n_pred)
#     loss_mat = torch.sum(torch.sum((I - torch.inverse(I + L)) * I, dim=4), dim=3)
#     mask = mask.masked_fill(torch.isnan(loss_mat), 0)
#     loss_mat = loss_mat.masked_fill(mask == 0, 0)
#     loss = -torch.sum(loss_mat) / torch.sum(mask)
#     return loss / 10


@jit.script
def multiNLL(y_pred, y_gt, mask, dim_feature=4, sig_penalisation=1):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    eps = 1e-1  # 10cm std is ok
    eps_rho = 1e-2

    y_gt = y_gt.unsqueeze(dim_feature - 1)
    repeat_idx = [1] * y_pred.dim()
    repeat_idx[dim_feature - 1] = y_pred.shape[dim_feature - 1]
    y_gt = y_gt.repeat((1, 1, 1, y_pred.shape[-2], 1))

    y_pred_pos = y_pred.narrow(dim_feature, 0, 2)
    muX = y_pred_pos.narrow(dim_feature, 0, 1)
    muY = y_pred_pos.narrow(dim_feature, 1, 1)
    sigX = torch.clamp(y_pred.narrow(dim_feature, 2, 1), eps, None)
    sigY = torch.clamp(y_pred.narrow(dim_feature, 3, 1), eps, None)
    rho = torch.clamp(y_pred.narrow(dim_feature, 4, 1), eps_rho - 1, 1 - eps_rho)
    p = y_pred.narrow(dim_feature, 5, 1)
    ohr = 1 / (1 - rho * rho)
    x = y_gt.narrow(dim_feature, 0, 1)
    y = y_gt.narrow(dim_feature, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    frac_x = diff_x / sigX
    frac_y = diff_y / sigY
    nll = 0.5 * ohr * (frac_x * frac_x + frac_y * frac_y -
                       2 * rho * frac_x * frac_y) + torch.log(sigX) + torch.log(sigY) - \
          0.5 * torch.log(ohr)

    ll = - nll + torch.log(p)
    ll = ll.squeeze(-1)
    nll = -logsumexp(ll, mask, dim=dim_feature - 1)

    if mask is None:
        lossVal = torch.mean(nll)
    else:
        nll = nll.masked_fill(mask == 0, 0)
        lossVal = torch.sum(nll) / torch.sum(mask)
    return lossVal


@jit.script
def multiNLLBest(y_pred, y_gt, mask, dim_feature=4, sig_penalisation=1):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    eps = 1e-1  # 10cm std is ok
    eps_rho = 1e-2

    y_gt = y_gt.unsqueeze(dim_feature - 1)
    repeat_idx = [1] * y_pred.dim()
    n_mix = y_pred.shape[-2]
    repeat_idx[dim_feature - 1] = y_pred.shape[dim_feature - 1]
    y_gt = y_gt.repeat((1, 1, 1, n_mix, 1))

    y_pred_pos = y_pred.narrow(dim_feature, 0, 2)
    muX = y_pred_pos.narrow(dim_feature, 0, 1)
    muY = y_pred_pos.narrow(dim_feature, 1, 1)
    sigX = torch.clamp(y_pred.narrow(dim_feature, 2, 1), eps, None)
    sigY = torch.clamp(y_pred.narrow(dim_feature, 3, 1), eps, None)
    rho = torch.clamp(y_pred.narrow(dim_feature, 4, 1), eps_rho - 1, 1 - eps_rho)
    p = y_pred.narrow(dim_feature, 5, 1)
    ohr = 1 / (1 - rho * rho)
    x = y_gt.narrow(dim_feature, 0, 1)
    y = y_gt.narrow(dim_feature, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    dist = diff_x[-1] * diff_x[-1] + diff_y[-1] * diff_y[-1]
    min_dist, arg_min_dist = torch.min(dist, dim=dim_feature - 2, keepdim=True)
    mask_best = (dist == min_dist).unsqueeze(0).repeat(p.shape[0], 1, 1, 1, 1)
    # arg_min_dist = arg_min_dist.unsqueeze(0).repeat(p.shape[0], 1, 1, 1, 1)
    frac_x = diff_x / sigX
    frac_y = diff_y / sigY
    nll = 0.5 * ohr * (frac_x * frac_x + frac_y * frac_y -
                       2 * rho * frac_x * frac_y) + torch.log(sigX) + torch.log(sigY) - \
          0.5 * torch.log(ohr)

    ll = - nll.detach() + torch.log(p)
    ll = ll.squeeze(-1)
    p_loss = -logsumexp(ll, mask, dim=dim_feature - 1)
    # p_loss = nn.CrossEntropyLoss()(p.view(-1, n_mix), arg_min_dist.view(-1))

    nll = nll.masked_fill(mask_best == 0, 0).squeeze(-1)
    nll = nll.sum(dim=dim_feature - 1)

    if mask is None:
        lossVal = torch.mean(nll)
        p_loss = torch.mean(p_loss)
    else:
        nll = nll.masked_fill(mask == 0, 0)
        lossVal = torch.sum(nll) / torch.sum(mask)
        p_loss = p_loss.masked_fill(mask == 0, 0)
        p_loss = torch.sum(p_loss) / torch.sum(mask)
    return lossVal + p_loss



@jit.script
def simpleADE(y_pred, y_gt, mask, dim=3):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum(torch.sqrt(diff_x*diff_x + diff_y*diff_y)*mask.unsqueeze(dim))/torch.sum(mask)
    else:
        output = torch.mean(torch.sqrt(diff_x*diff_x + diff_y*diff_y))
    return output


@jit.script
def simpleMSE(y_pred, y_gt, mask, dim=3):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum((diff_x*diff_x + diff_y*diff_y)*mask.unsqueeze(dim))/torch.sum(mask)
    else:
        output = torch.mean(diff_x*diff_x + diff_y*diff_y)
    return output

@jit.script
def multiMSE(y_pred, y_gt, mask, dim=4, p_ind=5):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim-1).repeat((1, 1, 1, y_pred.shape[3], 1))
    p = y_pred.narrow(dim, p_ind, 1)
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1)*mask.unsqueeze(dim-1))/torch.sum(mask)
    else:
        output = torch.mean(torch.sum(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1), 3))

    return output


@jit.script
def multiADE(y_pred, y_gt, mask, dim=4, p_ind=5):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim-1).repeat((1, 1, 1, y_pred.shape[3], 1))
    p = y_pred.narrow(dim, p_ind, 1)
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum(torch.sqrt(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1))*mask.unsqueeze(dim-1))/torch.sum(mask)
    else:
        output = torch.mean(torch.sqrt(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1)))

    return output


@jit.script
def multiFDE(y_pred, y_gt, mask, dim=4, p_ind=5):
    # type: (Tensor, Tensor, Tensor, int, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim-1).repeat((1, 1, 1, y_pred.shape[3], 1))
    p = y_pred.narrow(dim, p_ind, 1)
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x - muX
    diff_y = y - muY
    if mask is not None:
        output = torch.sum((torch.sqrt(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1))*mask.unsqueeze(dim-1))[-1]) / torch.sum(mask[-1])
    else:
        output = torch.mean(torch.sqrt(torch.sum((diff_x*diff_x + diff_y*diff_y)*p, dim-1))[-1])

    return output

@jit.script
def minADE(y_pred, y_gt, mask, dim=4):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim - 1)
    muX = y_pred.narrow(dim, 0, 1)[-1]
    muY = y_pred.narrow(dim, 1, 1)[-1]
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x[-1] - muX
    diff_y = y[-1] - muY

    dist = (diff_x*diff_x + diff_y*diff_y)
    _, min_fde = torch.min(dist, 2)

    min_fde = min_fde.unsqueeze(0).unsqueeze(dim).repeat(30, 1, 1, 1, 1)
    muX = torch.gather(y_pred.narrow(dim, 0, 1), 3, min_fde)
    muY = torch.gather(y_pred.narrow(dim, 1, 1), 3, min_fde)

    diff_x = x - muX
    diff_y = y - muY

    if mask is not None:
        mask = mask.unsqueeze(dim-1).unsqueeze(dim)
        output = torch.sum((torch.sqrt(diff_x*diff_x + diff_y*diff_y)*mask)) / torch.sum(mask)
    else:
        output = torch.mean(torch.sqrt(diff_x*diff_x + diff_y*diff_y))

    return output

@jit.script
def minFDE(y_pred, y_gt, mask, dim=4):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim - 1)
    muX = y_pred.narrow(dim, 0, 1)[-1]
    muY = y_pred.narrow(dim, 1, 1)[-1]
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x[-1] - muX
    diff_y = y[-1] - muY

    dist = (diff_x*diff_x + diff_y*diff_y)
    _, arg_min_fde = torch.min(dist, 2)
    fde = torch.gather(dist, 2, arg_min_fde.unsqueeze(dim-1))

    if mask is not None:
        mask = mask[-1].unsqueeze(dim-2).unsqueeze(dim-1)
        output = torch.sum(torch.sqrt(fde)*mask) / torch.sum(mask)
    else:
        output = torch.mean(torch.sqrt(fde))

    return output

@jit.script
def missRate(y_pred, y_gt, mask, dim=4):
    # type: (Tensor, Tensor, Tensor, int) -> Tensor
    y_gt = y_gt.unsqueeze(dim - 1)
    muX = y_pred.narrow(dim, 0, 1)
    muY = y_pred.narrow(dim, 1, 1)
    x = y_gt.narrow(dim, 0, 1)
    y = y_gt.narrow(dim, 1, 1)
    diff_x = x[-1] - muX[-1]
    diff_y = y[-1] - muY[-1]

    dist = (diff_x * diff_x + diff_y * diff_y)
    min_fde, arg_min_fde = torch.min(dist, 2)
    if mask is not None:
        mask = mask[-1].unsqueeze(dim - 2)
        output = torch.sum((min_fde > 4).float()*mask.float()) / torch.sum(mask.float())
    else:
        output = torch.mean((min_fde > 4))

    return output

