import torch

from models.MLP import *
from models.resnet import *
from models.vgg import *


FEATURE_MAP_SIZES = {1: (256, 56, 56),
                     2: (512, 28, 28),
                     3: (1024, 14, 14),
                     4: (2048, 7, 7)}


class LocalFeat(nn.Module):

    def __init__(self, input_channels, attention=False):
        super(LocalFeat, self).__init__()
        layers = [nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=1)]

        if attention:
            layers.append(nn.Sigmoid())

        self.extractor = nn.Sequential(*layers)

    def forward(self, x):
        # print('forward attention block')
        x = self.extractor(x)
        return x


class LocalFeatureModule(nn.Module):

    def __init__(self, args, in_channels):
        super(LocalFeatureModule, self).__init__()
        self.in_channels = in_channels
        if FEATURE_MAP_SIZES[1] in self.in_channels:
            self.local_block_1 = LocalFeat(256)
            self.fc_block_1 = nn.Sequential(nn.Linear(in_features=2 * (56 * 56), out_features=(56 * 56)), nn.ReLU())
        else:
            self.local_block_1 = None
            self.fc_block_1 = None

        if FEATURE_MAP_SIZES[2] in self.in_channels:
            self.local_block_2 = LocalFeat(512)
            self.fc_block_2 = nn.Sequential(nn.Linear(in_features=2 * (28 * 28), out_features=(28 * 28)), nn.ReLU())

        else:
            self.local_block_2 = None
            self.fc_block_2 = None

        if FEATURE_MAP_SIZES[3] in self.in_channels:
            self.local_block_3 = LocalFeat(1024)
            self.fc_block_3 = nn.Sequential(nn.Linear(in_features=2 * (14 * 14), out_features=(14 * 14)), nn.ReLU())

        else:
            self.local_block_3 = None
            self.fc_block_3 = None

        if FEATURE_MAP_SIZES[4] in self.in_channels:
            self.local_block_4 = LocalFeat(2048)
            self.fc_block_4 = nn.Sequential(nn.Linear(in_features=2 * (7 * 7), out_features=(7 * 7)), nn.ReLU())
        else:
            self.local_block_4 = None
            self.fc_block_4 = None

        self.layers = {256: self.local_block_1,
                       512: self.local_block_2,
                       1024: self.local_block_3,
                       2048: self.local_block_4}

        self.fc_blocks = {256: self.fc_block_1,
                          512: self.fc_block_2,
                          1024: self.fc_block_3,
                          2048: self.fc_block_4}

        self.merge_method = args.merge_method

        total_channels = 0
        for (C, H, W) in self.in_channels:
            total_channels += (H * W)  # todo residual connection?

        self.local_concat = nn.Sequential(nn.Linear(in_features=total_channels, out_features=2048), nn.ReLU())

        if not self.merge_method.startswith('local-diff-sim'):
            self.classifier = nn.Sequential(nn.Linear(in_features=2048, out_features=1))
        else:
            self.classifier = None

    def forward(self, x1s, x2s):
        rets = []
        # print('forward attention module')

        li_1s = []
        li_2s = []

        for (C, _, _), x1, x2 in zip(self.in_channels, x1s, x2s):
            li_1s.append(self.layers[C](x1).flatten(start_dim=1))
            li_2s.append(self.layers[C](x2).flatten(start_dim=1))

        ls = []

        for (C, _, _), l1, l2 in zip(self.in_channels, li_1s, li_2s):
            ls.append(self.fc_blocks[C](utils.vector_merge_function(l1, l2, method='concat')))

        # l1_1 = self.local_block_1(x1s[0]).flatten(start_dim=1)
        # l1_2 = self.local_block_1(x2s[0]).flatten(start_dim=1)
        #
        # l2_1 = self.local_block_2(x1s[1]).flatten(start_dim=1)
        # l2_2 = self.local_block_2(x2s[1]).flatten(start_dim=1)
        #
        # l3_1 = self.local_block_3(x1s[2]).flatten(start_dim=1)
        # l3_2 = self.local_block_3(x2s[2]).flatten(start_dim=1)
        #
        # l4_1 = self.local_block_4(x1s[3]).flatten(start_dim=1)
        # l4_2 = self.local_block_4(x2s[3]).flatten(start_dim=1)
        #
        # l1 = self.fc_block_1(utils.vector_merge_function(l1_1, l1_2, method='concat'))
        # l2 = self.fc_block_2(utils.vector_merge_function(l2_1, l2_2, method='concat'))
        # l3 = self.fc_block_3(utils.vector_merge_function(l3_1, l3_2, method='concat'))
        # l4 = self.fc_block_4(utils.vector_merge_function(l4_1, l4_2, method='concat'))

        local_features = self.local_concat(torch.cat(ls, dim=1))

        if not self.merge_method.startswith('local-diff-sim'):
            ret = self.classifier(local_features)
        else:
            ret = local_features

        return ret, local_features




class TopModel(nn.Module):

    def __init__(self, args, ft_net, sm_net, aug_mask=False, attention=False):
        super(TopModel, self).__init__()
        self.ft_net = ft_net
        print('ResNet50 parameters:', utils.get_number_of_parameters(self.ft_net))
        self.sm_net = sm_net
        self.aug_mask = aug_mask
        self.attention = attention
        self.merge_method = args.merge_method
        self.softmax = args.softmax_diff_sim
        self.fmaps_no = [int(i) for i in args.feature_map_layers]

        if self.merge_method.startswith('local'):
            feature_map_inputs = [FEATURE_MAP_SIZES[i] for i in self.fmaps_no]
            print(f'Using {feature_map_inputs} for local maps')
            self.local_features = LocalFeatureModule(args, feature_map_inputs)  # only for resnet50
        else:
            self.local_features = None

        if self.merge_method.startswith('local-diff-sim'):
            self.diffsim_fc_net = VectorConcat(input_size=4096,
                                               output_size=2048,
                                               layers=1)

            if self.merge_method.startswith('local-diff-sim-concat'):
                # in_feat = (56 * 56) + (28 * 28) + (14 * 14) + (7 * 7) + 4096
                in_feat = 2048 + 2048

            elif self.merge_method.startswith('local-diff-sim-mult') or self.merge_method.startswith(
                    'local-diff-sim-add'):
                # in_feat = (56 * 56) + (28 * 28) + (14 * 14) + (7 * 7) + 4096
                in_feat = 2048
            else:
                raise Exception(f"Local merge method not supported! {self.merge_method}")

            self.classifier = nn.Linear(in_features=in_feat, out_features=1)
        else:
            self.diffsim_fc_net = None
            self.classifier = None
            # if self.mask:
            #     self.input_layer = nn.Sequential(list(self.ft_net.children())[0])
            #     self.ft_net = nn.Sequential(*list(self.ft_net.children())[1:])
            #     # self.ft_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #     with torch.no_grad():
            #         with torch.no_grad():
            #             self.input_layer.weight[:, :3] = conv1weight
            # self.input_layer.weight[:, 3] = self.ft_net.conv1.weight[:, 0]

            # print('FEATURE NET')
            # print(self.ft_net)
            # print('SIAMESE NET')
            # print(self.sm_net)

    def get_activations_gradient(self):
        return self.ft_net.get_activations_gradient()

    # method for the activation exctraction
    def get_activations(self):
        return self.ft_net.get_activations()

    @utils.MY_DEC
    def forward(self, x1, x2, single=False, feats=False, dist=False, hook=False):
        # print('model input:', x1[-1].size())

        x1_f, x1_all = self.ft_net(x1, is_feat=True, hook=hook)
        if hook:
            anch_pass_act = self.get_activations().detach().clone()
        else:
            anch_pass_act = None
        out1, out2 = None, None

        if single and feats:
            raise Exception('Both single and feats cannot be True')

        if not single:
            x2_f, x2_all = self.ft_net(x2, is_feat=True, hook=hook)
            if hook:
                other_pass_act = self.get_activations().detach().clone()
            else:
                other_pass_act = None

            if self.merge_method.startswith('local'):
                x1_input = []
                x2_input = []

                for i in self.fmaps_no:
                    x1_input.append(x1_all[i - 1])
                    x2_input.append(x2_all[i - 1])

                ret, local_features = self.local_features(x1_input, x2_input)

                if self.merge_method.startswith('local-diff-sim'):  # TODO
                    ret_global = utils.vector_merge_function(x1_f, x2_f, method='diff-sim',
                                                             softmax=self.softmax).flatten(
                        start_dim=1)  # todo should be 2048 for now
                    ret_global = self.diffsim_fc_net(ret_global)

                    if self.merge_method.startswith('local-diff-sim-concat'):
                        final_vec = torch.cat([ret_global, ret], dim=1)

                    elif self.merge_method.startswith('local-diff-sim-add'):
                        final_vec = ret_global + ret

                    elif self.merge_method.startswith('local-diff-sim-mult'):
                        final_vec = ret_global * ret

                    else:
                        raise Exception(f"Local merge method not supported! {self.merge_method}")

                    pred = self.classifier(final_vec)


                else:
                    pred = ret

                if feats:
                    if hook:
                        return pred, local_features, x1_all, x2_all, [anch_pass_act, other_pass_act]
                    else:
                        return pred, local_features, x1_all, x2_all
                else:
                    return pred, local_features

            else:
                ret = self.sm_net(x1_f, x2_f, feats=feats, softmax=self.softmax)

                if feats:
                    pred, pdist, out1, out2 = ret
                    if hook:
                        return pred, pdist, out1, out2, [anch_pass_act, other_pass_act]
                    else:
                        return pred, pdist, out1, out2
                else:
                    pred, pdist = ret
                    return pred, pdist
        else:
            output = self.sm_net(x1_f, None, single)  # single is true
            return output

    def get_classifier_weights(self):
        return self.sm_net.get_classifier_weights()
        # print('features:', x2_f[-1].size())
        # print('output:', output.size())

        # if feats:
        #     return output, out1, out2
        # else:
        #     return output


def top_module(args, trained_feat_net=None, trained_sm_net=None, num_classes=1, mask=False, fourth_dim=False):
    if trained_sm_net is None:
        sm_net = MLP(args)
    else:
        sm_net = trained_sm_net

    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'vgg16': vgg16,
        'resnet101': resnet101,
    }

    use_pretrained = not (args.from_scratch)

    if trained_feat_net is None:
        print('Using pretrained model')
        ft_net = model_dict[args.feat_extractor](args, pretrained=use_pretrained, num_classes=num_classes, mask=mask,
                                                 fourth_dim=fourth_dim)
    else:
        print('Using recently trained model')
        ft_net = trained_feat_net

    if not args.freeze_ext:
        print("Unfreezing Resnet")
        for param in ft_net.parameters():
            param.requires_grad = True
    else:
        print("Freezing Resnet")
        for param in ft_net.parameters():
            param.requires_grad = False

    return TopModel(args=args, ft_net=ft_net, sm_net=sm_net, aug_mask=(mask and fourth_dim), attention=args.attention)
