import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


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


class BatchHard(nn.Module):
    # https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/triplet_loss.py

    def __init__(self, args, margin, soft=False):
        super(BatchHard, self).__init__()
        #
        # if margin > 1:
        #     raise Exception("Distances are normalized. Margine should be less than 1.0"
        self.margin = margin
        self.soft = soft

    def forward(self, batch, labels, get_idx=False):

        distances = utils.squared_pairwise_distances(batch)

        gpu = labels.device.type == 'cuda'

        mask_positive = utils.get_valid_positive_mask(labels, gpu)
        hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]
        hardest_positive_dist_idx = (distances * mask_positive.float()).max(dim=1)[1]

        mask_negative = utils.get_valid_negative_mask(labels, gpu)
        max_negative_dist = distances.max(dim=1, keepdim=True)[0]
        distances = distances + max_negative_dist * (~mask_negative).float()
        hardest_negative_dist = distances.min(dim=1)[0]
        hardest_negative_dist_idx = distances.min(dim=1)[1]

        if self.soft:
            loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        else:
            loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

        loss = loss.mean()

        if get_idx:
            return loss, (hardest_positive_dist_idx,
                          hardest_negative_dist_idx)
        else:
            return loss


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


class StopGradientLoss(nn.Module):
    def __init__(self, args):
        super(StopGradientLoss, self).__init__()

    def __calc_negative_cosine_sim(self, p, z):
        z = z.detach()  # stop gradient
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)

        return -(p * z).sum(dim=1).mean()


    def forward(self, p1, p2, z1, z2):  # negative cosine similarity
        loss1 = self.__calc_negative_cosine_sim(p1, z2)
        loss2 = self.__calc_negative_cosine_sim(p2, z1)

        return (loss1 + loss2) / 2


###
# TODO
# 1. Unit hemesphere
# 2. Dot product
# 3. Representations from ResNet rather than projection network
# 4. Augs?? AutoAug... ??
###
