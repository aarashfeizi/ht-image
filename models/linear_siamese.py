import torch.nn as nn
import utils
import torch.nn as nn

import utils


class LiSiamese(nn.Module):

    def __init__(self, args):
        super(LiSiamese, self).__init__()

        self.merge_method = args.merge_method

        if self.merge_method == 'diff-sim' or self.merge_method == 'concat':
            method_coefficient = 2
        else:
            method_coefficient = 1

        if args.feat_extractor == 'resnet50':
            self.input_shape = 2048
        elif args.feat_extractor == 'vgg16':
            self.input_shape = 4096
        else:
            self.input_shape = 512

        self.merge_input_shape = method_coefficient * self.input_shape

        self.extra_layer = args.extra_layer
        # self.layer = nn.Sequential(nn.Linear(25088, 512))
        layers = []
        input_size = self.merge_input_shape

        if args.dim_reduction != 0:
            self.dim_reduction_layer = nn.Sequential(nn.Linear(self.input_shape, args.dim_reduction),
                                                     nn.ReLU())
            input_size = args.dim_reduction * method_coefficient
        else:
            self.dim_reduction_layer = None

        if args.bn_before_classifier:
            layers.append(nn.BatchNorm1d(input_size))

        if self.extra_layer > 0:


            for i in range(self.extra_layer):

                if args.leaky_relu:
                    relu = nn.LeakyReLU(0.1)
                else:
                    relu = nn.ReLU()

                if args.static_size != 0:
                    layers.append(nn.Linear(input_size, args.static_size))
                    layers.append(relu)
                    if args.normalize:
                        layers.append(nn.BatchNorm1d(args.static_size))
                    input_size = args.static_size
                else:
                    layers.append(nn.Linear(input_size, input_size // 2))
                    layers.append(relu)
                    if args.normalize:
                        layers.append(nn.BatchNorm1d(input_size // 2))
                    input_size = input_size // 2

            # self.layer1 = nn.Sequential(*layers)

        layers.append(nn.Linear(input_size, 1))
            # self.layer2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

        # self.bn_for_classifier = nn.BatchNorm1d(self.input_shape)
        self.classifier = nn.Sequential(*layers)  # no sigmoid for bce_with_crossentorpy loss!!!!

        # return -1 if (a, n) and 1 if (a, p). Should learn a "distance function"
        # self.out = nn.Sequential(nn.Linear(self.input_shape, 1), nn.Tanh())

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)
        if self.dim_reduction_layer:
            x = self.dim_reduction_layer(x)
        # if self.extra_layer > 0:
        #     x = self.layer1(x)

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
