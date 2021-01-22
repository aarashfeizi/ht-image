import torch.nn as nn
import utils
import torch.nn as nn

import utils


class LiSiamese(nn.Module):

    def __init__(self, args):
        super(LiSiamese, self).__init__()

        self.merge_method = args.merge_method

        if self.merge_method == 'diff-sim':
            method_coefficient = 2
        else:
            method_coefficient = 1

        if args.feat_extractor == 'resnet50':
            self.input_shape = method_coefficient * 2048
        elif args.feat_extractor == 'vgg16':
            self.input_shape = method_coefficient * 4096
        else:
            self.input_shape = method_coefficient * 512

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

        # self.bn_for_classifier = nn.BatchNorm1d(self.input_shape)
        self.classifier = nn.Sequential(nn.Linear(self.input_shape, 1))  # no sigmoid for bce_with_crossentorpy loss!!!!
        # return -1 if (a, n) and 1 if (a, p). Should learn a "distance function"
        # self.out = nn.Sequential(nn.Linear(self.input_shape, 1), nn.Tanh())

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        if self.extra_layer > 0:
            x = self.layer1(x)

        # x = nn.functional.normalize(x, p=2, dim=1)
        # x = x / torch.norm(x)  # normalize to unit hypersphere

        return x

    def forward(self, x1, x2, single=False, feats=False):
        out1 = self.forward_one(x1)
        if single:
            return out1

        out2 = self.forward_one(x2)


        # out_cat = torch.cat((out1, out2), 1)
        # out_dist = torch.pow((out1 - out2), 2)
        out_dist = utils.vector_merge_function(out1, out2, method=self.merge_method)

        # out_dist = self.bn_for_classifier(out_dist)
        # dis = torch.abs(out1 - out2)
        pred = self.classifier(out_dist)  # output between -inf and inf. Passed through sigmoid in loss function

        # dis = torch.nn.PairwiseDistance()(out1, out2) / 2  # output between 0 and 1. 0 meaning similar and 1 meaning different

        #  return self.sigmoid(out)
        if feats:
            return pred, out_dist, out1, out2
        else:
            return pred, out_dist

    def get_classifier_weights(self):
        return self.classifier[0].weight
