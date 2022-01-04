import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


# todo between local features, use the nearest/farthest distances among them (between two image tensors) as distances of two tensors?'

def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T


class ProxyNCA_classic(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed) / 8)
        self.scale = scale

    def forward(self, X, T):
        P = self.proxies

        # note: self.scale is equal to sqrt(1/T)
        # in the paper T = 1/9, therefore, scale = sart(1/(1/9)) = sqrt(9) = 3
        #  we need to apply sqrt because the pairwise distance is calculated as norm^2
        P = self.scale * F.normalize(P, p=2, dim=-1)
        X = self.scale * F.normalize(X, p=2, dim=-1)

        D = utils.pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared=True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T=T, nb_classes=len(P), smoothing_const=0
        )
        loss1 = torch.sum(T * torch.exp(-D), -1)
        loss2 = torch.sum((1 - T) * torch.exp(-D), -1)
        loss = -torch.log(loss1 / loss2)
        loss = loss.mean()
        return loss


class LinkPredictionLoss(nn.Module):
    """
        Choose k nearest neighbors and calculate a BCE-Cross-Entropy loss
         on the anchor and each one of the k neighbors
    """

    def __init__(self, args, k=5, loss_metric='euclidean', temp=1):
        super(LinkPredictionLoss, self).__init__()
        self.k = k
        self.temperature = temp
        # self.bce_with_logit = torch.nn.BCEWithLogitsLoss()
        self.bce = torch.nn.BCELoss()
        self.metric = loss_metric
        self.emb = args.emb

    def forward(self, batch, labels):
        if self.emb:
            return self.forward_emb(batch, labels)
        else:
            return self.forward_bce(batch, labels)

    def forward_bce(self, batch, labels):
        # dot_product = torch.matmul(batch, batch.T)  # between -inf and +inf
        # min_value = dot_product.min().item() - 1
        # dot_product = dot_product.fill_diagonal_(min_value)

        # preds = F.sigmoid(dot_product) # between 0 and 1

        if self.metric == 'euclidean':
            euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

            preds = 2 * F.sigmoid(-euc_distances / self.temperature)  # between 0 and 1 (map inf to 0, and 0 to 1)

            sorted_indices = euc_distances.argsort()[:, :-1]

        else:
            batch = F.normalize(batch, p=2)
            cosine_sim = torch.matmul(batch, batch.T)  # between -1 and 1
            min_value = cosine_sim.min().item() - 1
            cosine_sim = cosine_sim.fill_diagonal_(min_value)

            preds = (cosine_sim + 1) / 2  # between 0 and 1

            sorted_indices = (-cosine_sim).argsort()[:, :-1]

        neighbor_indices_ = sorted_indices[:, :self.k]
        indices = torch.tensor([[j for _ in range(self.k)] for j in range(len(labels))])
        neighbor_preds_ = preds[indices, neighbor_indices_]
        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(self.k).view(-1, self.k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss = self.bce_with_logit(neighbor_preds_, true_labels)
        loss = self.bce(neighbor_preds_, true_labels)

        return loss

        # gpu = labels.device.type == 'cuda'
        #
        # mask_positive = utils.get_valid_positive_mask(labels, gpu)
        # pos_loss = ((1 - dot_product) * mask_positive.float()).sum(dim=1)
        # # positive_dist_idx = (cosine_sim * mask_positive.float())
        #
        # mask_negative = utils.get_valid_negative_mask(labels, gpu)
        # neg_loss = (F.relu(dot_product - self.margin) * mask_negative.float()).sum(dim=1)
        #
        # if self.l != 0:
        #     distances = utils.squared_pairwise_distances(batch)
        #     # import pdb
        #     # pdb.set_trace()
        #     idxs = distances.argsort()[:, 1]
        #     reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        # else:
        #     reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

    def forward_emb(self, batch, labels):
        euc_distances = utils.pairwise_distance(batch, diag_to_max=True)  # between 0 and inf

        euc_distances_sorted, sorted_indices = euc_distances.sort()


        sorted_indices = sorted_indices[:, :-1]
        sorted_euc_distances = euc_distances_sorted[:, :-1]

        k = min(self.k, sorted_euc_distances.shape[1])

        if k == 0:
            k = sorted_euc_distances.shape[1] # if k=0, consider the whole batch

        neighbor_indices_ = sorted_indices[:, :k]
        neighbor_distances_ = sorted_euc_distances[:, :k]

        neighbor_labels_ = labels[neighbor_indices_]

        true_labels = (neighbor_labels_ == labels.repeat_interleave(k).view(-1, k))  # boolean tensor
        true_labels = true_labels.type(torch.float32)

        # loss1 = torch.sum(true_labels * torch.exp(-neighbor_distances_), -1)
        # loss2 = torch.sum((1 - true_labels) * torch.exp(-neighbor_distances_), -1)
        # loss = -torch.log(loss1 / loss2)

        loss = torch.sum(-true_labels * F.log_softmax(-neighbor_distances_, dim=-1), dim=-1)

        loss = loss.mean()

        return loss


class LocalTripletLoss(nn.Module):
    def __init__(self, args, margin, soft=False):
        super(LocalTripletLoss, self).__init__()

        self.margin = margin
        self.loss = 0
        self.pd = torch.nn.PairwiseDistance(p=2)
        self.soft = soft

    def forward(self, anch_tensors, pos_tensor, neg_tensor, att_maps=None):
        if type(anch_tensors) == list:  # different anch activations for pos and neg
            posanch_tensor = anch_tensors[0]
            neganch_tensor = anch_tensors[1]

            N, C, H, W = posanch_tensor.size()

            posanch_tensor_tensor_locals = posanch_tensor.view(N, C, H * W).transpose(2, 1)
            neganch_tensor_tensor_locals = neganch_tensor.view(N, C, H * W).transpose(2, 1)
            pos_tensor_locals = pos_tensor.view(N, C, H * W).transpose(2, 1)
            neg_tensor_locals = neg_tensor.view(N, C, H * W).transpose(2, 1)

            posanch_tensor_tensor_locals = F.normalize(posanch_tensor_tensor_locals, dim=2)
            neganch_tensor_tensor_locals = F.normalize(neganch_tensor_tensor_locals, dim=2)
            pos_tensor_locals = F.normalize(pos_tensor_locals, dim=2)
            neg_tensor_locals = F.normalize(neg_tensor_locals, dim=2)

            pos_dist = torch.cdist(posanch_tensor_tensor_locals, pos_tensor_locals, p=2).min(axis=2)[0].sum(axis=1)
            neg_dist = torch.cdist(neganch_tensor_tensor_locals, neg_tensor_locals, p=2).min(axis=2)[0].sum(axis=1)

        else:  # same anch activations
            anch_tensor = anch_tensors
            N, C, H, W = anch_tensor.size()

            anch_tensor_locals = anch_tensor.view(N, C, H * W).transpose(2, 1)
            pos_tensor_locals = pos_tensor.view(N, C, H * W).transpose(2, 1)
            neg_tensor_locals = neg_tensor.view(N, C, H * W).transpose(2, 1)

            anch_tensor_locals = F.normalize(anch_tensor_locals, dim=2)
            pos_tensor_locals = F.normalize(pos_tensor_locals, dim=2)
            neg_tensor_locals = F.normalize(neg_tensor_locals, dim=2)

            if att_maps is not None:
                anch_map_flattened = att_maps[0].view(N, -1)
                pos_map_flattened = att_maps[1].view(N, -1)
                neg_map_flattened = att_maps[2].view(N, -1)
                pos_dist = (torch.cdist(anch_tensor_locals, pos_tensor_locals, p=2).min(axis=2)[0] * (
                        anch_map_flattened + pos_map_flattened)).sum(axis=1)
                neg_dist = (torch.cdist(anch_tensor_locals, neg_tensor_locals, p=2).min(axis=2)[0] * (
                        anch_map_flattened + neg_map_flattened)).sum(axis=1)
            else:
                pos_dist = torch.cdist(anch_tensor_locals, pos_tensor_locals, p=2).min(axis=2)[0].sum(axis=1)
                neg_dist = torch.cdist(anch_tensor_locals, neg_tensor_locals, p=2).min(axis=2)[0].sum(axis=1)

        if self.soft:
            loss = F.softplus(pos_dist - neg_dist)
        else:
            dist = pos_dist - neg_dist + self.margin
            loss = F.relu(dist)

        loss = loss.mean()

        return loss


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


class BatchAllGeneralization(nn.Module):
    # https://github.com/Yuol96/pytorch-triplet-loss/blob/master/model/triplet_loss.py

    def __init__(self, args, margin):
        super(BatchAllGeneralization, self).__init__()

        self.margin = margin

    def forward(self, batch, labels):
        distances = utils.squared_pairwise_distances(batch, sqrt=True)

        gpu = labels.device.type == 'cuda'

        mask_positive = utils.get_valid_positive_mask(labels, gpu)
        pos_loss = (distances * mask_positive.float()).exp().sum(dim=1).log()
        # positive_dist_idx = (cosine_sim * mask_positive.float())

        mask_negative = utils.get_valid_negative_mask(labels, gpu)
        neg_loss = ((self.margin - (distances * mask_negative.float()))).exp().sum(dim=1).log()

        loss = F.relu(pos_loss + neg_loss).sum()

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


class ContrastiveLoss(nn.Module):

    def __init__(self, args, margin, soft=False, l=0.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.l = l

    def forward(self, batch, labels):

        cosine_sim = torch.matmul(batch, batch.T)

        gpu = labels.device.type == 'cuda'

        mask_positive = utils.get_valid_positive_mask(labels, gpu)
        pos_loss = ((1 - cosine_sim) * mask_positive.float()).sum(dim=1)
        # positive_dist_idx = (cosine_sim * mask_positive.float())

        mask_negative = utils.get_valid_negative_mask(labels, gpu)
        neg_loss = (F.relu(cosine_sim - self.margin) * mask_negative.float()).sum(dim=1)

        if self.l != 0:
            distances = utils.squared_pairwise_distances(batch)
            # import pdb
            # pdb.set_trace()
            idxs = distances.argsort()[:, 1]
            reg_loss = -1 * distances.gather(1, idxs.view(-1, 1)).mean()
        else:
            reg_loss = None

        # if self.soft:
        #     loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        # else:
        #     loss = (hardest_positive_dist - hardest_negative_dist + self.margin).clamp(min=0)

        cont_loss = (pos_loss + neg_loss)

        cont_loss = cont_loss.mean()

        if reg_loss:
            loss = cont_loss + self.l * reg_loss
        else:
            loss = cont_loss
            reg_loss = torch.Tensor([0])

        return loss, cont_loss, reg_loss

###
# TODO
# 1. Unit hemesphere
# 2. Dot product
# 3. Representations from ResNet rather than projection network
# 4. Augs?? AutoAug... ??
###
