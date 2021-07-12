import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class VectorConcat(nn.Module):
    def __init__(self, input_size, output_size, layers=2, gated=False):
        super(VectorConcat, self).__init__()
        if layers < 1:
            raise Exception('Vector Concat Module should have at least 1 layer')

        concat_fc_layers = [nn.Linear(input_size, output_size), nn.ReLU()]

        for i in range(layers - 1):
            concat_fc_layers.append(nn.Linear(output_size, output_size))
            concat_fc_layers.append(nn.ReLU())

        if gated:
            concat_fc_layers[-1] = nn.Sigmoid()

        self.concat_fc = nn.Sequential(*concat_fc_layers)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = utils.vector_merge_function(x1, x2, method='concat')
        else:
            x = x1

        x = self.concat_fc(x)
        return x


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()

        self.merge_method = args.merge_method

        if self.merge_method == 'diff-sim' or \
                self.merge_method == 'concat':
            method_coefficient = 2
        elif self.merge_method == 'diff-sim-con-complete':
            method_coefficient = 4
        elif self.merge_method == 'diff-sim-con':
            method_coefficient = 2
        elif self.merge_method == 'diff-sim-con-att' or self.merge_method == 'diff-sim-con-att-add':
            method_coefficient = 1
        else:
            method_coefficient = 1

        if args.feat_extractor == 'resnet50':
            self.input_shape = 2048
        elif args.feat_extractor == 'deit16_224':
            self.input_shape = 768
        elif args.feat_extractor == 'vgg16':
            self.input_shape = 4096
        else:
            self.input_shape = 512

        self.merge_input_shape = method_coefficient * self.input_shape

        self.classifier_layer = args.classifier_layer
        self.projection_layer = args.projection_layer
        # self.layer = nn.Sequential(nn.Linear(25088, 512))
        layers = []
        input_size = self.merge_input_shape
        print(f'*#* 1. input_size = {input_size}')

        if args.dim_reduction != 0:
        #     self.dim_reduction_layer = nn.Sequential(nn.Linear(self.input_shape, args.dim_reduction),
        #                                              nn.ReLU())
            input_size = args.dim_reduction * method_coefficient
        # else:
        #     self.dim_reduction_layer = None

        if args.bn_before_classifier:
            layers.append(nn.BatchNorm1d(input_size))

        # self.attention = args.attention
        # if self.attention:
        #     input_size += 2048
        projection_layers = []
        if self.projection_layer > 0:
            feature_size = self.input_shape if args.dim_reduction == 0 else args.dim_reduction
            for i in range(self.projection_layer - 1):
                print(f'*#* feature_size = {feature_size}')
                projection_layers.append(nn.Linear(feature_size, feature_size))
                projection_layers.append(nn.ReLU())
                if args.normalize:
                    projection_layers.append(nn.BatchNorm1d(feature_size))

            projection_layers.append(nn.Linear(feature_size, feature_size))
            if args.normalize:
                projection_layers.append(nn.BatchNorm1d(feature_size))

            self.projector = nn.Sequential(*projection_layers)
        else:
            self.projector = None

        if self.classifier_layer > 0:
            for i in range(self.classifier_layer):

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
                    print(f'*#* input_size = {input_size}')
                    layers.append(nn.Linear(input_size, input_size // 2))
                    layers.append(relu)
                    if args.normalize:
                        layers.append(nn.BatchNorm1d(input_size // 2))
                    input_size = input_size // 2

        if self.merge_method == 'diff-sim-con' or \
                self.merge_method == 'diff-sim-con-att' or \
                self.merge_method == 'diff-sim-con-att-add':

            # if args.dim_reduction != 0:
            #     att_size = args.dim_reduction * 2
            #
            # else:
            #
            att_size = self.input_shape * 2


            self.concat_fc_net = VectorConcat(input_size=int(att_size),
                                              output_size=int(att_size / 2),
                                              layers=args.att_extra_layer,
                                              gated=self.merge_method == 'diff-sim-con-att')

            self.diffsim_fc_net = VectorConcat(input_size=int(att_size),
                                               output_size=int(att_size / 2),
                                               layers=1)
        else:
            self.concat_fc_net = None
            self.diffsim_fc_net = None

        # self.layer1 = nn.Sequential(*layers)

        layers.append(nn.Linear(input_size, 1))
        # self.layer2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())

        # self.bn_for_classifier = nn.BatchNorm1d(self.input_shape)
        self.classifier = nn.Sequential(*layers)  # no sigmoid for bce_with_crossentorpy loss!!!!

        # return -1 if (a, n) and 1 if (a, p). Should learn a "distance function"
        # self.out = nn.Sequential(nn.Linear(self.input_shape, 1), nn.Tanh())

    def forward_one(self, x):
        x = x.view(x.size()[0], -1)

        if self.projector:
            x = self.projector(x)

        x = F.normalize(x, p=2, dim=1)

        # if self.dim_reduction_layer:
        #     x = self.dim_reduction_layer(x)

        # x = nn.functional.normalize(x, p=2, dim=1)
        # x = x / torch.norm(x)  # normalize to unit hypersphere

        return x

    def forward(self, x1, x2, single=False, feats=False, mid_att=None, softmax=False):
        out1 = self.forward_one(x1)
        if single:
            return out1

        out2 = self.forward_one(x2)

        # out_cat = torch.cat((out1, out2), 1)
        # out_dist = torch.pow((out1 - out2), 2)
        out_dist = utils.vector_merge_function(out1, out2, method=self.merge_method, softmax=softmax)

        if self.concat_fc_net:

            out_dist = self.diffsim_fc_net(out_dist)

            att = self.concat_fc_net(out1, out2)
            if self.merge_method == 'diff-sim-con':
                out_dist = utils.vector_merge_function(out_dist, att, method='concat')
            elif self.merge_method == 'diff-sim-con-att':
                out_dist = out_dist * att
            elif self.merge_method == 'diff-sim-con-att-add':
                out_dist = out_dist + att


        if self.merge_method == 'diff-sim-con-complete':
            concat = utils.vector_merge_function(out1, out2, method='concat')
            out_dist = utils.vector_merge_function(out_dist, concat, method='concat')
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

class StopGrad_MLP(nn.Module):
    def __init__(self, args):
        super(StopGrad_MLP, self).__init__()

        if args.feat_extractor == 'resnet50':
            self.input_shape = 2048
        elif args.feat_extractor == 'vgg16':
            self.input_shape = 4096
        else:
            self.input_shape = 512

        self.projection_layer = args.projection_layer
        if args.dim_reduction != 0:
            self.input_shape = args.dim_reduction

        layers = []
        for i in range(0, self.projection_layer - 1):
            layers += [nn.Linear(self.input_shape, self.input_shape),
                       nn.ReLU()]
            if args.normalize:
                layers += [nn.BatchNorm1d(self.input_shape)]

        layers += [nn.Linear(self.input_shape, self.input_shape)]

        if args.normalize:
            layers += [nn.BatchNorm1d(self.input_shape)]

        self.projection = nn.Sequential(*layers)

        self.bottleneck = nn.Sequential(nn.Linear(self.input_shape, self.input_shape // 4),
                                        nn.ReLU(), nn.BatchNorm1d(self.input_shape // 4),
                                        nn.Linear(self.input_shape // 4, self.input_shape))

    def forward(self, x, y, single=False):
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        x_pred = self.bottleneck(x)

        if single:
            return x_pred

        y = y.view(x.size()[0], -1)
        y = self.projection(y)

        y_pred = self.bottleneck(y)

        return x, y, x_pred, y_pred
