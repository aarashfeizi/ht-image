import torch
import torch.nn as nn
from torch.nn import Parameter
import utils


class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return utils.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return utils.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


# class GeMmp(nn.Module):
#
#     def __init__(self, p=3, mp=1, eps=1e-6):
#         super(GeMmp, self).__init__()
#         self.p = Parameter(torch.ones(mp) * p)
#         self.mp = mp
#         self.eps = eps
#
#     def forward(self, x):
#         return utils.gem(x, p=self.p.unsqueeze(-1).unsqueeze(-1), eps=self.eps)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' + 'p=' + '[{}]'.format(self.mp) + ', ' + 'eps=' + str(self.eps) + ')'
#

class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return utils.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'

