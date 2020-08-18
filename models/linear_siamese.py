import torch
import torch.nn as nn


class LiSiamese(nn.Module):

    def __init__(self, args):
        super(LiSiamese, self).__init__()

        if args.feat_extractor == 'resnet50':
            self.input_shape = 2048
        else:
            self.input_shape = 512

        self.extra_layer = args.extra_layer
        # self.layer = nn.Sequential(nn.Linear(25088, 512))
        if self.extra_layer > 0:
            layers = []
            for i in range(self.extra_layer):
                layers.append(nn.Linear(self.input_shape, self.input_shape))
                layers.append(nn.ReLU())
                if args.normalize:
                    layers.append(nn.BatchNorm1d(self.input_shape))

            self.layer1 = nn.Sequential(*layers)

            # self.layer2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(2 * self.input_shape, 1))  # no sigmoid for bce_with_crossentorpy loss!!!!

        # return -1 if (a, n) and 1 if (a, p). Should learn a "distance function"
        # self.out = nn.Sequential(nn.Linear(self.input_shape, 1), nn.Tanh())

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        if self.extra_layer > 0:
            x = self.layer1(x)

        # x = x / torch.norm(x)  # normalize to unit hypersphere

        return x

    def forward(self, x1, x2, single=False, feats=False, dist=False):
        out1 = self.forward_one(x1)
        if single:
            return out1

        out2 = self.forward_one(x2)


        out_cat = torch.cat((out1, out2), 1)

        # dis = torch.abs(out1 - out2)

        dis = self.out(out_cat)  # output between -inf and inf. Passed through sigmoid in loss function

        # dis = torch.nn.PairwiseDistance()(out1, out2) / 2  # output between 0 and 1. 0 meaning similar and 1 meaning different

        #  return self.sigmoid(out)
        if feats:
            return dis, out1, out2
        else:
            return dis
