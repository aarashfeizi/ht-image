import torch
import torch.nn.functional as F

from models.MLP import *
from models.resnet import *
from models.vgg import *

FEATURE_MAP_SIZES = {1: (256, 56, 56),
                     2: (512, 28, 28),
                     3: (1024, 14, 14),
                     4: (2048, 7, 7)}


# https://github.com/SaoYan/LearnToPayAttention/

class LinearAttentionBlock_Spatial(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock_Spatial, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

    def forward(self, l, g):
        N, C, W, H = l.size()
        if g is not None:
            c = self.op(l + g)  # batch_sizex1xWxH
        else:
            c = self.op(l)

        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        l_att = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            l_att = l_att.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            l_att = F.adaptive_avg_pool2d(l_att, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return a, l_att


class LinearAttentionBlock_Channel(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock_Channel, self).__init__()
        self.normalize_attn = normalize_attn

    def forward(self, l1, l2):
        N, C, W, H = l1.size()
        if l2 is not None:
            c = l1 + l2  # batch_sizex1xWxH
            c = torch.sum(c, dim=(2, 3))
        else:
            c = l1

        if self.normalize_attn:
            a = F.softmax(c, dim=1).view(N, C, 1, 1)
        else:
            a = torch.sigmoid(c)

        a = torch.mul(a.expand_as(l1), l1)
        if self.normalize_attn:
            l_att = a.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            l_att = F.adaptive_avg_pool2d(a, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return a, l_att


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


class Projector(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Projector, self).__init__()
        self.op = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)

    def forward(self, x):
        x = self.op(x)
        return x


class LocalFeatureModule(nn.Module):

    def __init__(self, args, in_channels, global_dim):
        super(LocalFeatureModule, self).__init__()
        self.in_channels = in_channels
        self.merge_global = args.merge_global
        self.global_dim = global_dim
        self.no_global = args.no_global
        self.global_attention = not args.local_to_local

        if args.spatial_att:
            att_module = LinearAttentionBlock_Spatial
        else:
            att_module = LinearAttentionBlock_Channel

        if FEATURE_MAP_SIZES[1] in self.in_channels:
            self.projector1 = Projector(256, global_dim) if self.global_dim != 256 else None
            # self.fc_block_1 = nn.Sequential(nn.Linear(in_features=2 * (56 * 56), out_features=(56 * 56)), nn.ReLU())
            self.att_1 = att_module(global_dim)
        else:
            self.projector1 = None
            # self.fc_block_1 = None
            self.att_1 = None

        if FEATURE_MAP_SIZES[2] in self.in_channels:
            self.projector2 = Projector(512, global_dim) if self.global_dim != 512 else None
            # self.fc_block_2 = nn.Sequential(nn.Linear(in_features=2 * (28 * 28), out_features=(28 * 28)), nn.ReLU())
            self.att_2 = att_module(global_dim)
        else:
            self.projector2 = None
            # self.fc_block_2 = None
            self.att_2 = None

        if FEATURE_MAP_SIZES[3] in self.in_channels:
            self.projector3 = Projector(1024, global_dim) if self.global_dim != 1024 else None
            # self.fc_block_3 = nn.Sequential(nn.Linear(in_features=2 * (14 * 14), out_features=(14 * 14)), nn.ReLU())
            self.att_3 = att_module(global_dim)
        else:
            self.projector3 = None
            # self.fc_block_3 = None
            self.att_3 = None

        if FEATURE_MAP_SIZES[4] in self.in_channels:

            self.projector4 = Projector(2048, global_dim) if self.global_dim != 2048 else None
            self.att_4 = att_module(global_dim)
        else:
            self.projector4 = None
            # self.fc_block_4 = None
            self.att_4 = None


        self.layers = {256: self.projector1,
                       512: self.projector2,
                       1024: self.projector3,
                       2048: self.projector4}

        # self.fc_blocks = {256: self.fc_block_1,
        #                   512: self.fc_block_2,
        #                   1024: self.fc_block_3,
        #                   2048: self.fc_block_4}

        self.atts = {256: self.att_1,
                     512: self.att_2,
                     1024: self.att_3,
                     2048: self.att_4}

        self.merge_method = args.merge_method

        total_channels = 0
        for (C, H, W) in self.in_channels:
            total_channels += (H * W)  # todo residual connection?

        if self.merge_method == 'local-ds-attention':
            self.att_merge = 'diff-sim'
            coeff = 2
        elif self.merge_method == 'local-attention':
            self.att_merge = 'sim'
            coeff = 1
        else:
            self.att_merge = ''
            coeff = -1  # to raise an error in case it is usesd

        # self.local_concat = nn.Sequential(nn.Linear(in_features=total_channels, out_features=2048), nn.ReLU())

        if not self.merge_method.startswith('local-diff-sim'):
            self.classifier = nn.Sequential(nn.Linear(in_features=global_dim * len(in_channels) * coeff, out_features=1))
        else:
            self.classifier = None


    def __project(self, x_local):
        lis = []

        for (C, _, _), x in zip(self.in_channels, x_local):
            # if C != 2048:
            #     li_1s.append(self.layers[C](x1).flatten(start_dim=1))
            #     li_2s.append(self.layers[C](x2).flatten(start_dim=1))
            # else:
            #     li_1s.append(x1.flatten(start_dim=1))
            #     li_2s.append(x2.flatten(start_dim=1))
            if C != self.global_dim:
                lis.append(self.layers[C](x))
            else:
                lis.append(x)

        return lis


    def __attend_to_locals(self, loc_feat, glob_feat, glob_feat_2=None):

        atts = []
        att_gs = []

        for (C, _, _), l in zip(self.in_channels, loc_feat):
            if self.no_global or (glob_feat is None):
                att, att_g = self.atts[C](l, None)

            elif self.merge_global:
                att, att_g = self.atts[C](l, glob_feat + glob_feat_2)

            else:
                att, att_g = self.atts[C](l, glob_feat)

            atts.append(att)
            att_gs.append(att_g)

        return atts, att_gs

    def __attend_to_local_w_local(self, loc_feat, loc_feat2):

        atts = []
        att_gs = []

        if loc_feat2 is not None:
            for (C, _, _), l1, l2 in zip(self.in_channels, loc_feat, loc_feat2):
                att, att_g = self.atts[C](l1, l2)

                atts.append(att)
                att_gs.append(att_g)
        else:
            for (C, _, _), l1 in zip(self.in_channels, loc_feat):
                att, att_g = self.atts[C](l1, None)

                atts.append(att)
                att_gs.append(att_g)

        return atts, att_gs


    def forward(self, x1_local, x2_local=None, x1_global=None, x2_global=None, single=False):
        rets = []
        # print('forward attention module')
        li_1s = self.__project(x1_local)

        if not single:
            li_2s = self.__project(x2_local)
        else:
            li_2s = None

        if self.global_attention:
            atts_1, att_gs_1 = self.__attend_to_locals(li_1s, x1_global, glob_feat_2=x2_global)
        else:
            atts_1, att_gs_1 = self.__attend_to_local_w_local(li_1s, li_2s)

        if not single:
            if self.global_attention:
                atts_2, att_gs_2 = self.__attend_to_locals(li_2s, x2_global, glob_feat_2=x1_global)
            else:
                atts_2, att_gs_2 = self.__attend_to_local_w_local(li_2s, li_1s)
        else:
            att_gs_2 = []
            atts_2 = []

        att_gs = []

        if not single:
            for (att_g_1, att_g_2) in zip(att_gs_1, att_gs_2):
                att_gs.append(utils.vector_merge_function(att_g_1, att_g_2, method=self.att_merge))
        else:
            att_gs = att_gs_1

        local_features = torch.cat(att_gs, dim=1)

        if self.classifier and not single:
            ret = self.classifier(local_features)
        else:
            ret = local_features

        if not single:
            return ret, local_features, atts_1, atts_2, torch.cat(att_gs_1, dim=1), torch.cat(att_gs_2, dim=1)
        else:
            return ret, local_features, atts_1, torch.cat(att_gs_1, dim=1) # only local_features is important, ret does not make any sense

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

        if args.feat_extractor == 'resnet50':
            ft_net_output = 2048
        elif args.feat_extractor == 'vgg16':
            ft_net_output = 4096
        else:
            ft_net_output = 512

        if args.dim_reduction != 0:
            ft_net_output = args.dim_reduction

        if self.merge_method.startswith('local'):
            feature_map_inputs = [FEATURE_MAP_SIZES[i] for i in self.fmaps_no]
            print(f'Using {feature_map_inputs} for local maps')
            self.local_features = LocalFeatureModule(args, feature_map_inputs, global_dim=ft_net_output)  # only for resnet50

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
    def forward(self, x1, x2, single=False, feats=False, dist=False, hook=False, return_att=False):
        # print('model input:', x1[-1].size())
        atts_1 = None
        atts_2 = None
        x1_global, x1_local = self.ft_net(x1, is_feat=True, hook=hook)

        if hook:
            anch_pass_act = self.get_activations().detach().clone()
        else:
            anch_pass_act = None
        out1, out2 = None, None

        if single and feats:
            raise Exception('Both single and feats cannot be True')

        if not single:
            x2_global, x2_local = self.ft_net(x2, is_feat=True, hook=hook)
            if hook:
                other_pass_act = self.get_activations().detach().clone()
            else:
                other_pass_act = None

            if self.merge_method.startswith('local'):
                x1_input = []
                x2_input = []

                for i in self.fmaps_no:
                    x1_input.append(x1_local[i - 1])
                    x2_input.append(x2_local[i - 1])

                ret, local_features, atts_1, atts_2, att_x1_local, att_x2_local = self.local_features(x1_local=x1_input,
                                                                                                      x2_local=x2_input,
                                                                                                      x1_global=x1_global,
                                                                                                      x2_global=x2_global)


                if self.merge_method.startswith('local-diff-sim'):  # TODO

                    ret_global = utils.vector_merge_function(x1_global, x2_global, method='diff-sim',
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
                        if return_att:
                            return pred, local_features, att_x1_local, att_x2_local, [anch_pass_act,
                                                                              other_pass_act], atts_1, atts_2
                        else:
                            return pred, local_features, att_x1_local, att_x2_local, [anch_pass_act, other_pass_act]
                    else:
                        return pred, local_features, att_x1_local, att_x2_local
                else:
                    return pred, local_features

            else:
                ret = self.sm_net(x1_global, x2_global, feats=feats, softmax=self.softmax)

                if feats:
                    pred, pdist, out1, out2 = ret
                    if hook:
                        if return_att:
                            return pred, pdist, out1, out2, [anch_pass_act, other_pass_act], atts_1, atts_2
                        else:
                            return pred, pdist, out1, out2, [anch_pass_act, other_pass_act]
                    else:
                        return pred, pdist, out1, out2
                else:
                    pred, pdist = ret
                    return pred, pdist
        else:
            if self.merge_method.startswith('local'):
                x1_input = []
                x2_input = []

                for i in self.fmaps_no:
                    x1_input.append(x1_local[i - 1])

                _, output, _, _ = self.local_features(x1_local=x1_input,
                                                                              x2_local=None,
                                                                              x1_global=x1_global,
                                                                              x2_global=None,
                                                                              single=single)

            else:
                output = self.sm_net(x1_global, None, single)  # single is true

            return output

    def get_classifier_weights(self):
        if self.classifier:
            return self.classifier.weight
        elif self.local_features and self.local_features.classifier:
            return self.local_features.classifier[0].weight
        else:
            return self.sm_net.get_classifier_weights()
        # print('features:', x2_f[-1].size())
        # print('output:', output.size())

        # if feats:
        #     return output, out1, out2
        # else:self.draw_activations
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
                                                 fourth_dim=fourth_dim, output_dim=args.dim_reduction)
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
