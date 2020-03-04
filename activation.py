import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.jit as jit


class base_Mish(nn.Module):
    def __init__(self):
        super(base_Mish, self).__init__()
    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class base_GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        #return torch.nn.functional.gelu(x.float())
        # The first approximation has more operations than the second
        # See https://arxiv.org/abs/1606.08415
        #return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * torch.sigmoid(1.702 * x)

Mish = torch.jit.script(base_Mish())
GELU = torch.jit.script(base_GELU())
