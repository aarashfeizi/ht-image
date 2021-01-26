import torch.nn as nn
import torch.nn.functional as F
import torch


class TripletLoss(nn.Module):
    def __init__(self, args, margin, soft=False):
        super(TripletLoss, self).__init__()
        #
        # if margin > 1:
        #     raise Exception("Distances are normalized. Margine should be less than 1.0")

        self.margin = margin
        self.no_negative = args.no_negative
        self.loss = 0
        self.pd = torch.nn.PairwiseDistance(p=2)
        self.soft = soft

    def forward(self, pos_dist, neg_dist):
        # pos_dist = self.pd(anch, pos)
        # neg_dist = self.pd(anch, neg)

        if self.soft:
            loss = F.softplus(pos_dist - neg_dist)
        else:
            dist = pos_dist - neg_dist + self.margin
            loss = F.relu(dist)

        loss = loss.mean()

        return loss

class HardBatch(nn.Module):
    def __init__(self, args, margin, soft=False):
        super(TripletLoss, self).__init__()
        #
        # if margin > 1:
        #     raise Exception("Distances are normalized. Margine should be less than 1.0"
        self.margin = margin
        self.soft = soft


    def forward(self, batch, mask):

        return 0



class MaxMarginLoss(nn.Module):
    def __init__(self, args, margin):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        self.loss = 0
        self.pd = torch.nn.PairwiseDistance(p=2)


    def forward(self, pos_dist, neg_dist):

        # pos_dist = self.pd(anch, pos)
        # neg_dist = self.pd(anch, neg)

        neg_part = F.relu(self.margin - neg_dist)
        pos_part = pos_dist

        loss = neg_part.mean() + pos_part.mean()

        return loss, [pos_part, neg_part]

###
# TODO
# 1. Unit hemesphere
# 2. Dot product
# 3. Representations from ResNet rather than projection network
# 4. Augs?? AutoAug... ??
###
