import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.MLP import *
from models.resnet import *
from models.vgg import *
from models.deit import *

from torch.utils.data import DataLoader
from my_datasets import Local_Feat_Dataset

FEATURE_MAP_SIZES = {1: (256, 56, 56),
                     2: (512, 28, 28),
                     3: (1024, 14, 14),
                     4: (2048, 7, 7)}

# https://github.com/SaoYan/LearnToPayAttention/
A_SUM = [0, 0]


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

        if self.normalize_attn:  # todo plot "a" for "att_all"
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = c

        if H == 1:
            A_SUM[0] += torch.sum(a, dim=0).cpu().data.numpy()
            if type(N) != int:
                A_SUM[1] += N.cpu().data.numpy()
            else:
                A_SUM[1] += N

        l_att = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            l_att_vector = l_att.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            l_att_vector = F.adaptive_avg_pool2d(l_att, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return l_att, l_att_vector


class LinearAttentionBlock_Spatial2(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(LinearAttentionBlock_Spatial2, self).__init__()
        self.normalize_attn = normalize_attn
        self.op_transform = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0,
                                      bias=False)
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

        if constant_weight is not None:
            self.op_transform.weight.data.fill_(constant_weight)
            self.op.weight.data.fill_(constant_weight)

    def forward(self, l, g):
        N, C, W, H = l.size()
        if g is not None:
            g = self.op_transform(g)
            c = self.op(g)  # batch_sizex1xWxH
        else:
            g = self.op_transform(l)
            c = self.op(g)

        if self.normalize_attn:  # todo plot "a" for "att_all"
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = c

        l_att = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            l_att_vector = l_att.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            l_att_vector = F.adaptive_avg_pool2d(l_att, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return l_att, l_att_vector


class CrossDotProductAttentionBlock(nn.Module):
    def __init__(self, in_features, constant_weight=None):
        super(CrossDotProductAttentionBlock, self).__init__()

        self.layernorm = nn.LayerNorm([in_features, 7, 7])

        self.op_k = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
        self.op_q = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
        self.op_v = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)

        # self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)

        if constant_weight is not None:
            self.op_k.weight.data.fill_(constant_weight)
            self.op_k.weight.data.fill_(constant_weight)

            self.op_q.weight.data.fill_(constant_weight)
            self.op_q.weight.data.fill_(constant_weight)

            self.op_v.weight.data.fill_(constant_weight)
            self.op_v.weight.data.fill_(constant_weight)

    def forward(self, pre_local_query_org, pre_local_key_org):
        N, C, W, H = pre_local_query_org.size()

        pre_local_query = self.layernorm(pre_local_query_org)
        local_1_query = self.op_q(pre_local_query).reshape(N, C, W * H)

        if pre_local_key_org is not None:
            pre_local_key = self.layernorm(pre_local_key_org)
            local_2_key = self.op_k(pre_local_key).reshape(N, C, W * H)
        else:
            local_2_key = local_1_query

        attention_map = local_1_query.transpose(-2, -1) @ local_2_key

        query_atts_map = attention_map.sum(axis=2).softmax(axis=1).reshape(N, 1, W, H)
        key_atts_map = attention_map.sum(axis=1).softmax(axis=1).reshape(N, 1, W, H)

        attended_local1_from2 = (self.op_v(pre_local_key_org).reshape(N, C, W * H) @ attention_map.softmax(axis=1)).reshape(
            N, C, W, H)  # todo not sure if key should be multiplied or query

        attended_local1_from1 = (
                    attention_map.softmax(axis=2) @ self.op_v(pre_local_query_org).reshape(N, C, W * H).transpose(-2,
                                                                                                              -1)).reshape(
            N, C, W, H)

        attended_local1_asq = torch.mul(query_atts_map.expand_as(pre_local_query_org), pre_local_query_org)
        attended_local2_ask = torch.mul(key_atts_map.expand_as(pre_local_key_org), pre_local_key_org)

        attended_local1_asq = attended_local1_asq.view(N, C, -1).sum(dim=2)  # batch_sizexC
        attended_local2_ask = attended_local2_ask.view(N, C, -1).sum(dim=2)  # batch_sizexC

        attended_local1 = (pre_local_query_org +
                           attended_local1_from1 +
                           attended_local1_from2) # todo works because 2 additions

        # return c.view(N, 1, W, H), g
        return attended_local1_asq, attended_local2_ask, (query_atts_map, key_atts_map), attended_local1


class CrossDotProductAttention(nn.Module):
    def __init__(self, in_features, constant_weight=None, mode='query',
                 cross_add=False):  # mode can be "query", "key", and "both"
        super(CrossDotProductAttention, self).__init__()
        self.mode = mode
        self.cross_add = cross_add
        self.qk_module = CrossDotProductAttentionBlock(in_features=in_features, constant_weight=constant_weight)

    def return_representation(self, q, k):
        if self.mode == 'query':
            return q
        elif self.mode == 'key':
            return k
        elif self.mode == 'both':
            return (q + k) / 2
        else:
            raise Exception('Unsuppored representation generation for CrossDotProductAttention')

    def forward(self, local1, local2):

        if local2 is not None:
            attended_local_1_asq, attended_local_2_ask, (l1_query_map, l2_key_map), attended_local_1 = self.qk_module(
                local1, local2)
            attended_local_2_asq, attended_local_1_ask, (l2_query_map, l1_key_map), attended_local_2 = self.qk_module(
                local2, local1)
        else:
            attended_local_1_asq, attended_local_1_ask, (l1_query_map, l1_key_map), attended_local_1 = self.qk_module(
                local1, local1)
            return self.return_representation(attended_local_1_asq, attended_local_1_ask), None, \
                   (self.return_representation(l1_query_map, l1_key_map),
                    None)
        if self.cross_add:
            return attended_local_1, \
                   attended_local_2, \
                   (self.return_representation(l1_query_map, l1_key_map),
                    self.return_representation(l2_query_map, l2_key_map))
        else:
            return self.return_representation(attended_local_1_asq, attended_local_1_ask) + local1, \
                   self.return_representation(attended_local_2_asq, attended_local_2_ask) + local2, \
                   (self.return_representation(l1_query_map, l1_key_map),
                    self.return_representation(l2_query_map, l2_key_map))


class LinearAttentionBlock_Channel(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(LinearAttentionBlock_Channel, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)

        if constant_weight is not None:
            self.op.weight.data.fill_(constant_weight)

    def forward(self, l1, l2):
        N, C, W, H = l1.size()
        if l2 is not None:
            # c = l1 + l2  # batch_sizex1xWxH
            c = torch.sum(l2, dim=(2, 3)).view(N, C, 1, 1)
            c = self.op(c)
        else:
            c = torch.sum(l1, dim=(2, 3)).view(N, C, 1, 1)
            c = self.op(c)

        if self.normalize_attn:
            a = F.softmax(c.view(N, C), dim=1).view(N, C, 1, 1)
        else:
            a = torch.sigmoid(c)

        a = torch.mul(a.expand_as(l1), l1)
        if self.normalize_attn:
            l_att_vector = a.view(N, C, -1).sum(dim=2)  # batch_sizexC
        else:
            l_att_vector = F.adaptive_avg_pool2d(a, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return a, l_att_vector


class LinearAttentionBlock_Channel2(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(LinearAttentionBlock_Channel2, self).__init__()
        self.normalize_attn = normalize_attn
        # self.op = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
        self.op = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features))
        # self.op = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features // 4),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=in_features // 4, out_features=in_features))
        if constant_weight is not None:
            self.op.weight.data.fill_(constant_weight)

    def forward(self, l1, l2):
        N, C = l1.size()
        if l2 is not None:
            # c = l1 + l2  # batch_sizex1xWxH
            c = self.op(l2)
        else:
            c = self.op(l1)

        if self.normalize_attn:
            a = F.softmax(c.view(N, C), dim=1)
        else:
            a = torch.sigmoid(c)

        a = torch.mul(a.expand_as(l1), l1)
        l_att_vector = a.view(N, C)
        # if self.normalize_attn:
        #     l_att_vector = a.view(N, C, -1).sum(dim=2)  # batch_sizexC
        # else:
        #     l_att_vector = F.adaptive_avg_pool2d(a, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return a, l_att_vector


class LongRangedAttention(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(LongRangedAttention, self).__init__()
        self.normalize_attn = normalize_attn
        self.op1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
        # self.op2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)
        # self.op = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features))
        # self.op = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features // 4),
        #                         nn.ReLU(),
        #                         nn.Linear(in_features=in_features // 4, out_features=in_features))
        if constant_weight is not None:
            torch.nn.init.constant_(self.op1.weight, constant_weight)

    def forward(self, l1_org, l2_org):
        N, C, H, W = l1_org.size()

        l1 = self.op1(l1_org)
        l2 = self.op1(l2_org)

        l1_reshaped = l1.view(N, C, H * W, 1)
        l2_reshaped = l2.view(N, C, H * W, 1)

        A = torch.matmul(l1_reshaped, l2_reshaped.transpose(2, 3))

        A = F.softmax(A, dim=3)  # size (N, C, H*W, H*W) and normalized on last dim

        l1_forattention = l1.view(N, C, 1, H * W)
        l2_forattention = l2.view(N, C, 1, H * W)

        att_mask_from_l1 = torch.matmul(A.transpose(2, 3), l1_forattention.transpose(2, 3))  # size (N, C, H*W, 1)
        att_mask_from_l1 = F.softmax(att_mask_from_l1, dim=3).view(N, C, H, W)

        att_mask_from_l2 = torch.matmul(A, l2_forattention.transpose(2, 3))  # size (N, C, H*W, 1)
        att_mask_from_l2 = F.softmax(att_mask_from_l2, dim=3).view(N, C, H, W)

        l1_res = (l1_org * att_mask_from_l1)
        l1_att_vector = F.adaptive_avg_pool2d(l1_res, (1, 1)).view(N, C)
        att_mask_from_l1 = att_mask_from_l1.sum(dim=1)

        l2_res = (l2_org * att_mask_from_l2)
        l2_att_vector = F.adaptive_avg_pool2d(l2_res, (1, 1)).view(N, C)
        att_mask_from_l2 = att_mask_from_l2.sum(dim=1)

        # if self.normalize_attn:
        #     l_att_vector = a.view(N, C, -1).sum(dim=2)  # batch_sizexC
        # else:
        # l_att_vector = F.adaptive_avg_pool2d(a, (1, 1)).view(N, C)
        # return c.view(N, 1, W, H), g
        return [att_mask_from_l1, l1_att_vector], [att_mask_from_l2, l2_att_vector]


class LinearAttentionBlock_BOTH(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock_BOTH, self).__init__()
        self.normalize_attn = normalize_attn
        self.channel = LinearAttentionBlock_Channel(in_features)
        self.spatial = LinearAttentionBlock_Spatial(in_features)

    def forward(self, l1, l2, g1):
        l1, _ = self.channel.forward(l1, l2)

        l1_map, l1_vector = self.spatial.forward(l1, g1)

        return l1_map, l1_vector


class LinearAttentionBlock_GlbChannelSpatial(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(LinearAttentionBlock_GlbChannelSpatial, self).__init__()
        self.normalize_attn = normalize_attn
        self.spatial = LinearAttentionBlock_Spatial2(in_features,
                                                     constant_weight=constant_weight)  # transforms second one before applying it
        self.channel = LinearAttentionBlock_Channel2(in_features, constant_weight=constant_weight)

    def forward(self, g1, g2):
        N, C, W, H = g1.size()
        g1_map, g1_vector = self.spatial.forward(g1, g1)
        g1_vector, _ = self.channel.forward(g1_vector, g2.view(N, C))

        return g1_map, g1_vector


class Att_For_Unet(nn.Module):
    def __init__(self, in_features, normalize_attn=True, constant_weight=None):
        super(Att_For_Unet, self).__init__()
        self.normalize_attn = normalize_attn
        self.att = LongRangedAttention(in_features,
                                       constant_weight=constant_weight)  # transforms second one before applying it

    def forward(self, g1, g2):
        N, C, W, H = g1.size()
        [g1_map, g1_vector], [g2_map, g2_vector] = self.att.forward(g1, g2)

        return [g1_map, g1_vector], [g2_map, g2_vector]


class AttentionModule_C(nn.Module):
    def __init__(self, in_features, reduce_space=True):
        super(AttentionModule_C, self).__init__()
        if reduce_space:
            self.op = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=5, padding=0, bias=False)
        else:
            self.op = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, padding=0, bias=False)

    def forward(self, X):
        N, C, W, H = X.size()
        X = self.op(X)
        return X.reshape(N, C, -1)


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

    def __init__(self, input_channels, output_channels, maxpool=0):
        super(Projector, self).__init__()
        self.op = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1)
        if maxpool != 0:
            self.pool = nn.MaxPool2d(maxpool)
        else:
            self.pool = None

    def forward(self, x):
        x = self.op(x)
        if self.pool:
            x = self.pool(x)
        return x


class ChannelWiseAttention(nn.Module):

    def __init__(self, args, in_channels, global_dim):
        super(ChannelWiseAttention, self).__init__()
        self.in_channels = in_channels
        self.global_dim = global_dim
        self.cross_attention = args.cross_attention
        self.att_mode_sc = args.att_mode_sc

        if FEATURE_MAP_SIZES[1] in self.in_channels:
            self.projector1 = Projector(256, global_dim) if self.global_dim != 256 else None
            self.k_op1 = AttentionModule_C(global_dim)
            self.q_op1 = AttentionModule_C(global_dim)
            self.v_op1 = AttentionModule_C(global_dim, reduce_space=False)
            # self.fc_block_1 = nn.Sequential(nn.Linear(in_features=2 * (56 * 56), out_features=(56 * 56)), nn.ReLU())
        else:
            self.projector1 = None
            self.k_op1 = None
            self.q_op1 = None
            self.v_op1 = None
            # self.fc_block_1 = None
            self.att_1 = None

        if FEATURE_MAP_SIZES[2] in self.in_channels:
            self.projector2 = Projector(512, global_dim) if self.global_dim != 512 else None
            self.k_op2 = AttentionModule_C(global_dim)
            self.q_op2 = AttentionModule_C(global_dim)
            self.v_op2 = AttentionModule_C(global_dim, reduce_space=False)
            # self.fc_block_2 = nn.Sequential(nn.Linear(in_features=2 * (28 * 28), out_features=(28 * 28)), nn.ReLU())
        else:
            self.projector2 = None
            self.k_op2 = None
            self.q_op2 = None
            self.v_op2 = None
            # self.fc_block_2 = None

        if FEATURE_MAP_SIZES[3] in self.in_channels:
            self.projector3 = Projector(1024, global_dim) if self.global_dim != 1024 else None
            self.k_op3 = AttentionModule_C(global_dim)
            self.q_op3 = AttentionModule_C(global_dim)
            self.v_op3 = AttentionModule_C(global_dim, reduce_space=False)
            # self.fc_block_3 = nn.Sequential(nn.Linear(in_features=2 * (14 * 14), out_features=(14 * 14)), nn.ReLU())
        else:
            self.projector3 = None
            self.k_op3 = None
            self.q_op3 = None
            self.v_op3 = None
            # self.fc_block_3 = None

        if FEATURE_MAP_SIZES[4] in self.in_channels:

            self.projector4 = Projector(2048, global_dim) if self.global_dim != 2048 else None
            self.k_op4 = AttentionModule_C(global_dim)
            self.q_op4 = AttentionModule_C(global_dim)
            self.v_op4 = AttentionModule_C(global_dim, reduce_space=False)
        else:
            self.projector4 = None
            self.k_op4 = None
            self.q_op4 = None
            self.v_op4 = None

        self.layers = {256: self.projector1,
                       512: self.projector2,
                       1024: self.projector3,
                       2048: self.projector4}

        # self.fc_blocks = {256: self.fc_block_1,
        #                   512: self.fc_block_2,
        #                   1024: self.fc_block_3,
        #                   2048: self.fc_block_4}

        self.atts_k = {256: self.k_op1,
                       512: self.k_op2,
                       1024: self.k_op3,
                       2048: self.k_op4}

        self.atts_q = {256: self.q_op1,
                       512: self.q_op2,
                       1024: self.q_op3,
                       2048: self.q_op4}

        self.atts_v = {256: self.v_op1,
                       512: self.v_op2,
                       1024: self.v_op3,
                       2048: self.v_op4}

        self.merge_method = args.merge_method

        total_channels = 0
        for (C, H, W) in self.in_channels:
            total_channels += (H * W)  # todo residual connection?

        # self.local_concat = nn.Sequential(nn.Linear(in_features=total_channels, out_features=2048), nn.ReLU())

        if not self.merge_method.startswith('local-diff-sim'):
            self.classifier = nn.Sequential(
                nn.Linear(in_features=global_dim * len(in_channels), out_features=1))
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

    def __get_keys_querie_values(self, X):
        keys = []
        queries = []
        values = []

        for (C, _, _), f in zip(self.in_channels, X):
            keys.append(self.atts_k[C](f))
            queries.append(self.atts_q[C](f))
            values.append(self.atts_v[C](f))

        return keys, queries, values

    def __attend_to_locals(self, X, X2=None):
        K, Q, V = self.__get_keys_querie_values(X)
        if X2 is not None:
            _, Q2, _ = self.__get_keys_querie_values(X2)
            Qs = Q2
        else:
            Qs = Q

        attentions = []
        output = []
        for k, q, v, x in zip(K, Qs, V, X):
            N, C, W, H = x.size()
            attention_logits = torch.matmul(q, k.transpose(1, 2))  # (N, C, C)
            att = F.softmax(attention_logits, dim=1)
            attended_local = torch.matmul(att, v).reshape(N, C, W, H)
            attentions.append(attended_local)
            attended_local += x
            attended_global = F.adaptive_avg_pool2d(attended_local, (1, 1)).view(N, C)
            output.append(attended_global)

        return attentions, output

    def forward(self, x1_local, x2_local=None, single=False):
        rets = []
        # print('forward attention module')
        li_1s = self.__project(x1_local)

        if not single:
            li_2s = self.__project(x2_local)
        else:
            li_2s = None

        if (self.cross_attention and not single):
            atts_1, att_gs_1 = self.__attend_to_locals(li_1s, li_2s)
        else:
            atts_1, att_gs_1 = self.__attend_to_locals(li_1s, None)

        if not single:
            if self.cross_attention:
                atts_2, att_gs_2 = self.__attend_to_locals(li_2s, li_1s)
            else:
                atts_2, att_gs_2 = self.__attend_to_locals(li_2s, None)
        else:
            att_gs_2 = []
            atts_2 = []

        att_gs = []

        att_gs_1 = [F.normalize(v, p=2, dim=1) for v in att_gs_1]
        att_gs_2 = [F.normalize(v, p=2, dim=1) for v in att_gs_2]

        if not single:
            for (att_g_1, att_g_2) in zip(att_gs_1, att_gs_2):
                att_gs.append(utils.vector_merge_function(att_g_1, att_g_2, method='sim', normalize=False))
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
            return ret, local_features, atts_1, torch.cat(att_gs_1,
                                                          dim=1)  # only local_features is important, ret does not make any sense


class LocalFeatureModule(nn.Module):

    def __init__(self, args, in_channels, global_dim):
        super(LocalFeatureModule, self).__init__()
        self.in_channels = in_channels
        self.merge_global = args.merge_global
        self.global_dim = global_dim
        self.no_global = args.no_global
        self.global_attention = not args.local_to_local
        self.att_mode_sc = args.att_mode_sc
        self.spatial_projection = args.spatial_projection
        if args.att_on_all:
            self.att_all = LinearAttentionBlock_Spatial(global_dim)
        else:
            self.att_all = None

        if args.att_on_all:
            self.importance = nn.Parameter(torch.ones(len(in_channels), global_dim))
            self.att_att = None
        else:
            self.importance = None

        if self.spatial_projection:
            maxpools = [8, 4, 2, 0]
        else:
            maxpools = [0, 0, 0, 0]

        # spatial_att
        if args.att_mode_sc == 'both':
            att_module = LinearAttentionBlock_BOTH
        elif args.att_mode_sc == 'spatial':
            att_module = LinearAttentionBlock_Spatial
        else:  # args.att_mode_sc == 'channel'
            att_module = LinearAttentionBlock_Channel

        if FEATURE_MAP_SIZES[1] in self.in_channels:
            self.projector1 = Projector(256, global_dim, maxpool=maxpools[0]) if self.global_dim != 256 else None
            # self.fc_block_1 = nn.Sequential(nn.Linear(in_features=2 * (56 * 56), out_features=(56 * 56)), nn.ReLU())
            self.att_1 = att_module(global_dim)
        else:
            self.projector1 = None
            # self.fc_block_1 = None
            self.att_1 = None

        if FEATURE_MAP_SIZES[2] in self.in_channels:
            self.projector2 = Projector(512, global_dim, maxpool=maxpools[1]) if self.global_dim != 512 else None
            # self.fc_block_2 = nn.Sequential(nn.Linear(in_features=2 * (28 * 28), out_features=(28 * 28)), nn.ReLU())
            self.att_2 = att_module(global_dim)
        else:
            self.projector2 = None
            # self.fc_block_2 = None
            self.att_2 = None

        if FEATURE_MAP_SIZES[3] in self.in_channels:
            self.projector3 = Projector(1024, global_dim, maxpool=maxpools[2]) if self.global_dim != 1024 else None
            # self.fc_block_3 = nn.Sequential(nn.Linear(in_features=2 * (14 * 14), out_features=(14 * 14)), nn.ReLU())
            self.att_3 = att_module(global_dim)
        else:
            self.projector3 = None
            # self.fc_block_3 = None
            self.att_3 = None

        if FEATURE_MAP_SIZES[4] in self.in_channels:

            self.projector4 = Projector(2048, global_dim, maxpool=maxpools[3]) if self.global_dim != 2048 else None
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

        if self.spatial_projection:
            vecotrs_to_merge = 1
        else:
            vecotrs_to_merge = len(in_channels)

        # self.local_concat = nn.Sequential(nn.Linear(in_features=total_channels, out_features=2048), nn.ReLU())

        if not self.merge_method.startswith('local-diff-sim'):
            self.classifier = nn.Sequential(
                nn.Linear(in_features=global_dim * vecotrs_to_merge * coeff, out_features=1))
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

        if self.att_all is not None:
            num = len(att_gs)
            att_gs, _ = self.att_all(torch.stack(att_gs, dim=2).unsqueeze(dim=3), None)
            att_gs = att_gs.squeeze(dim=3)
            att_gs = torch.chunk(att_gs, num, dim=2)

            to_ret = []
            for a in att_gs:
                to_ret.append(a.squeeze(dim=2))

            att_gs = to_ret

        return atts, att_gs

    def __attend_to_local_w_local(self, loc_feat, loc_feat2, x1_global=None):

        atts = []
        att_gs = []

        if loc_feat2 is not None:
            for (C, _, _), l1, l2 in zip(self.in_channels, loc_feat, loc_feat2):
                if x1_global is not None:
                    att, att_g = self.atts[C](l1, l2, x1_global)
                else:
                    att, att_g = self.atts[C](l1, l2)

                atts.append(att)
                att_gs.append(att_g)
        else:
            for (C, _, _), l1 in zip(self.in_channels, loc_feat):
                if x1_global is not None:
                    att, att_g = self.atts[C](l1, None, x1_global)
                else:
                    att, att_g = self.atts[C](l1, None)

                atts.append(att)
                att_gs.append(att_g)

        return atts, att_gs

    def forward(self, x1_local, x2_local=None, x1_global=None, x2_global=None, single=False):
        rets = []
        # print('forward attention module')
        li_1s = self.__project(x1_local)

        if self.spatial_projection:
            li_1s = [torch.cat(li_1s, dim=3)]

        if not single:
            li_2s = self.__project(x2_local)
            if self.spatial_projection:
                li_2s = [torch.cat(li_2s, dim=3)]
        else:
            li_2s = None

        if self.att_mode_sc == 'both':

            if not single:
                atts_1, att_gs_1 = self.__attend_to_local_w_local(li_1s, li_2s, x1_global)
                atts_2, att_gs_2 = self.__attend_to_local_w_local(li_2s, li_1s, x2_global)
            else:
                atts_1, att_gs_1 = self.__attend_to_local_w_local(li_1s, None, x1_global)
                att_gs_2 = []
                atts_2 = []


        else:
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

        if self.spatial_projection:
            atts_1 = torch.chunk(atts_1[0], len(x1_local), dim=3)
            if not single:
                atts_2 = torch.chunk(atts_2[0], len(x2_local), dim=3)

        att_gs = []

        if self.importance is not None:
            att_gs_1 = [F.softmax(self.importance[i], dim=0) * v for i, v in enumerate(att_gs_1)]
            att_gs_2 = [F.softmax(self.importance[i], dim=0) * v for i, v in enumerate(att_gs_2)]

        att_gs_1 = [F.normalize(v, p=2, dim=1) for v in att_gs_1]
        att_gs_2 = [F.normalize(v, p=2, dim=1) for v in att_gs_2]

        if not single:
            for (att_g_1, att_g_2) in zip(att_gs_1, att_gs_2):
                att_gs.append(utils.vector_merge_function(att_g_1, att_g_2, method=self.att_merge, normalize=False))
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
            return ret, local_features, atts_1, torch.cat(att_gs_1,
                                                          dim=1)  # only local_features is important, ret does not make any sense


class DiffSimFeatureAttention(nn.Module):

    def __init__(self, args, in_channels, global_dim):
        super(DiffSimFeatureAttention, self).__init__()
        self.in_channels = in_channels
        self.global_dim = global_dim
        self.add_features = args.add_local_features

        # spatial_att

        att_module = LinearAttentionBlock_Spatial

        if FEATURE_MAP_SIZES[1] in self.in_channels:
            self.projector1 = Projector(256, global_dim)
            # self.fc_block_1 = nn.Sequential(nn.Linear(in_features=2 * (56 * 56), out_features=(56 * 56)), nn.ReLU())
            self.logit_1 = att_module(global_dim, normalize_attn=False)
        else:
            self.projector1 = None
            # self.fc_block_1 = None
            self.logit_1 = None

        if FEATURE_MAP_SIZES[2] in self.in_channels:
            self.projector2 = Projector(512, global_dim)
            # self.fc_block_2 = nn.Sequential(nn.Linear(in_features=2 * (28 * 28), out_features=(28 * 28)), nn.ReLU())
            self.logit_2 = att_module(global_dim, normalize_attn=False)
        else:
            self.projector2 = None
            # self.fc_block_2 = None
            self.logit_2 = None

        if FEATURE_MAP_SIZES[3] in self.in_channels:
            self.projector3 = Projector(1024, global_dim)
            # self.fc_block_3 = nn.Sequential(nn.Linear(in_features=2 * (14 * 14), out_features=(14 * 14)), nn.ReLU())
            self.logit_3 = att_module(global_dim, normalize_attn=False)
        else:
            self.projector3 = None
            # self.fc_block_3 = None
            self.logit_3 = None

        if FEATURE_MAP_SIZES[4] in self.in_channels:
            self.projector4 = Projector(2048, global_dim)
            self.logit_4 = att_module(global_dim, normalize_attn=False)
        else:
            self.projector4 = None
            # self.fc_block_4 = None
            self.logit_4 = None

        self.layers = {256: self.projector1,
                       512: self.projector2,
                       1024: self.projector3,
                       2048: self.projector4}

        self.logits = {256: self.logit_1,
                       512: self.logit_2,
                       1024: self.logit_3,
                       2048: self.logit_4}

        total_channels = 0
        for (C, H, W) in self.in_channels:
            total_channels += (H * W)  # todo residual connection?

        vectors_to_merge = len(in_channels)

        if self.add_features:
            vectors_to_merge = 1

        self.attention = nn.Sequential(
            nn.Linear(in_features=global_dim * vectors_to_merge, out_features=global_dim),
            nn.Softmax())

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

    def __attend_to_locals(self, loc_feat):

        atts = []
        att_gs = []

        for (C, _, _), l in zip(self.in_channels, loc_feat):
            att, att_g = self.logits[C](l, None)

            atts.append(att)
            att_gs.append(att_g)

        return atts, att_gs

    def forward(self, x1_local, x1_global, x2_local=None, x2_global=None, single=False):

        # print('forward attention module')
        li_1s = self.__project(x1_local)

        if not single:
            li_2s = self.__project(x2_local)
        else:
            li_2s = None

        atts_1, att_gs_1 = self.__attend_to_locals(li_1s)
        att_gs_1 = [F.normalize(v, p=2, dim=1) for v in att_gs_1]
        if self.add_features:
            local_att_1 = sum(att_gs_1)
        else:
            local_att_1 = torch.cat(att_gs_1, dim=1)
        attention_1 = self.attention(local_att_1)
        glb1 = (attention_1 * x1_global.squeeze(dim=-1).squeeze(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)

        if not single:
            atts_2, att_gs_2 = self.__attend_to_locals(li_2s)
            att_gs_2 = [F.normalize(v, p=2, dim=1) for v in att_gs_2]
            if self.add_features:
                local_att_2 = sum(att_gs_2)
            else:
                local_att_2 = torch.cat(att_gs_2, dim=1)

            attention_2 = self.attention(local_att_2)
            glb2 = (attention_2 * x2_global.squeeze(dim=-1).squeeze(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        else:
            return glb1, None

        return glb1, glb2


class TopModel(nn.Module):

    def __init__(self, args, ft_net, sm_net, aug_mask=False, attention=False):
        super(TopModel, self).__init__()
        self.ft_net = ft_net
        self.transformer = 'deit' in args.feat_extractor
        print('ResNet50 parameters:', utils.get_number_of_parameters(self.ft_net))
        self.sm_net = sm_net
        self.aug_mask = aug_mask
        self.attention = attention
        self.merge_method = args.merge_method
        self.softmax = args.softmax_diff_sim
        self.loss = args.loss
        self.fmaps_no = [int(i) for i in args.feature_map_layers]

        if args.feat_extractor == 'resnet50':
            ft_net_output = 2048
        elif args.feat_extractor == 'vgg16':
            ft_net_output = 4096
        else:
            ft_net_output = 512

        if args.dim_reduction != 0:
            ft_net_output = args.dim_reduction

        self.channel_attention = None
        self.local_features = None
        self.diffsim_fc_net = None
        self.no_final_network = args.no_final_network
        self.classifier = None
        self.attention_module = None
        self.global_attention = args.attention
        self.glb_atn = None
        self.att_type = ''
        if args.loss != 'stopgrad':
            if self.merge_method.startswith('local'):
                feature_map_inputs = [FEATURE_MAP_SIZES[i] for i in self.fmaps_no]
                print(f'Using {feature_map_inputs} for local maps')
                self.local_features = LocalFeatureModule(args, feature_map_inputs,
                                                         global_dim=ft_net_output)  # only for resnet50

            elif self.merge_method.startswith('channel-attention'):
                feature_map_inputs = [FEATURE_MAP_SIZES[i] for i in self.fmaps_no]
                print(f'Using {feature_map_inputs} for local maps')
                self.channel_attention = ChannelWiseAttention(args, feature_map_inputs,
                                                              global_dim=ft_net_output)

            elif self.merge_method.startswith('local-diff-sim'):
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

            elif (self.merge_method.startswith('diff') or self.merge_method.startswith('sim')) and self.global_attention:
                feature_map_inputs = [FEATURE_MAP_SIZES[i] for i in self.fmaps_no]
                self.attention_module = DiffSimFeatureAttention(args, feature_map_inputs, global_dim=ft_net_output)

            elif (self.merge_method.startswith('diff') or self.merge_method.startswith('sim')) and args.att_mode_sc == 'glb-both':
                self.att_type = 'channel_spatial'
                self.glb_atn = LinearAttentionBlock_GlbChannelSpatial(in_features=ft_net_output,
                                                                      constant_weight=args.att_weight_init)
            elif (self.merge_method.startswith('diff') or self.merge_method.startswith('sim')) and args.att_mode_sc == 'unet-att':
                self.att_type = 'unet'
                self.glb_atn = Att_For_Unet(in_features=ft_net_output, constant_weight=args.att_weight_init)
            elif (self.merge_method.startswith('diff') or self.merge_method.startswith('sim')) and args.att_mode_sc == 'dot-product':
                self.att_type = 'dot-product'
                self.glb_atn = CrossDotProductAttention(in_features=ft_net_output,
                                                        constant_weight=args.att_weight_init,
                                                        mode=args.dp_type,
                                                        cross_add=False)

            elif (self.merge_method.startswith('diff') or self.merge_method.startswith('sim')) and args.att_mode_sc == 'dot-product-add':
                self.att_type = 'dot-product-add'
                self.glb_atn = CrossDotProductAttention(in_features=ft_net_output,
                                                        constant_weight=args.att_weight_init,
                                                        mode=args.dp_type,
                                                        cross_add=True)

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

    def forward(self, x1, x2, single=False, feats=False, dist=False, hook=False, return_att=False):
        # print('model input:', x1[-1].size())

        if self.transformer:
            x1_global = self.ft_net(x1)
            x1_local = [None]
        else:
            x1_global, x1_local = self.ft_net(x1, is_feat=True, hook=hook)

        if single and feats:
            raise Exception('Both single and feats cannot be True')

        if not single:

            if self.transformer:
                x2_global = self.ft_net(x2)
                x2_local = [None]
            else:
                x2_global, x2_local = self.ft_net(x2, is_feat=True, hook=hook)

            results = self.classify(globals=[x1_global, x2_global], locals=[x1_local, x2_local],
                                    feats=feats, hook=hook, return_att=return_att)
            return results
        else:  # single

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

                output = F.normalize(output, p=2, dim=1)
                # import pdb
                # pdb.set_trace()

                # output = x1_global.squeeze()
            elif self.merge_method.startswith('channel-attention'):
                x1_input = []
                x2_input = []

                for i in self.fmaps_no:
                    x1_input.append(x1_local[i - 1])

                _, output, _, _ = self.channel_attention(
                    x1_local=x1_input,
                    x2_local=None,
                    single=single)

                output = F.normalize(output, p=2, dim=1)
            else:

                if self.global_attention:
                    x1_input = []

                    for i in self.fmaps_no:
                        x1_input.append(x1_local[i - 1])

                    x1_global, _ = self.attention_module(x1_input, x1_global, None, None, single=single)
                if self.att_type == 'channel_spatial':
                    # print('Using glb_atn! *********')
                    _, x1_global = self.glb_atn(x1_local[-1], x1_global)
                elif self.att_type == 'dot-product' or self.att_type == 'dot-product-add':
                    attended_x1_global, _, (x1_map, _) = self.glb_atn(x1_local[-1], None)
                    x1_global = attended_x1_global
                elif self.att_type == 'unet':
                    pass  # shouldn't pass through attention

                if self.no_final_network:
                    x1_global = x1_global.view((x1_global.size()[0], -1))
                    x1_global = F.normalize(x1_global, p=2, dim=1)
                    output = x1_global
                else:
                    output = self.sm_net(x1_global, None, single)  # single is true



            return output, x1_local[-1]

    def classify(self, globals, locals, feats=True, hook=False, return_att=False):

        x1_global, x2_global = globals
        x1_local, x2_local = locals
        atts_1 = None
        atts_2 = None
        if hook:
            anch_pass_act = [l.detach().clone() for l in x1_local]
        else:
            anch_pass_act = None
        out1, out2 = None, None
        if hook:
            other_pass_act = [l.detach().clone() for l in x2_local]
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

            att_x1_local = F.normalize(att_x1_local, p=2, dim=1)
            att_x2_local = F.normalize(att_x2_local, p=2, dim=1)

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
        elif self.merge_method.startswith('channel-attention'):
            x1_input = []
            x2_input = []

            for i in self.fmaps_no:
                x1_input.append(x1_local[i - 1])
                x2_input.append(x2_local[i - 1])

            ret, local_features, atts_1, atts_2, att_x1_local, att_x2_local = self.channel_attention(
                x1_local=x1_input,
                x2_local=x2_input)

            pred = ret

            att_x1_local = F.normalize(att_x1_local, p=2, dim=1)
            att_x2_local = F.normalize(att_x2_local, p=2, dim=1)

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
        else:  # diff, sim, or diff-sim
            if self.loss == 'stopgrad':
                x1, x2, x1_pred, x2_pred = self.sm_net(x1_global, x2_global) # todo doesn't support args.no_final_network

                return x1, x2, x1_pred, x2_pred
            else:
                attended_x1_global = None
                attended_x2_global = None

                if self.global_attention:
                    x1_input = []
                    x2_input = []

                    for i in self.fmaps_no:
                        x1_input.append(x1_local[i - 1])
                        x2_input.append(x2_local[i - 1])

                    x1_global, x2_global = self.attention_module(x1_input, x1_global, x2_input, x2_global)

                if self.att_type == 'channel_spatial':
                    # print('Using glb_atn! *********')
                    _, attended_x1_global = self.glb_atn(x1_local[-1], x2_global)
                    _, attended_x2_global = self.glb_atn(x2_local[-1], x1_global)

                    x1_global = x1_global.squeeze(dim=-1).squeeze(dim=-1) + attended_x1_global
                    x2_global = x2_global.squeeze(dim=-1).squeeze(dim=-1) + attended_x2_global
                elif self.att_type == 'unet':
                    [_, attended_x1_global], [_, attended_x2_global] = self.glb_atn(x1_local[-1], x2_local[-1])
                    x1_global = x1_global.squeeze(dim=-1).squeeze(dim=-1) + attended_x1_global
                    x2_global = x2_global.squeeze(dim=-1).squeeze(dim=-1) + attended_x2_global
                elif self.att_type == 'dot-product' or self.att_type == 'dot-product-add':
                    N, C, H, W = x1_local[-1].size()
                    attended_x1_global, attended_x2_global, (atts_1, atts_2) = self.glb_atn(x1_local[-1],
                                                                                            x2_local[-1])
                    # utils.save_representation_hists(attended_x1_global, savepath='attentions.npy')
                    # utils.save_representation_hists(attended_x2_global, savepath='attentions.npy')
                    # utils.save_representation_hists(x1_global.squeeze(dim=-1).squeeze(dim=-1), savepath='realglobals.npy')
                    # utils.save_representation_hists(x1_global.squeeze(dim=-1).squeeze(dim=-1), savepath='realglobals.npy')
                    # x1_global = F.normalize(x1_global.squeeze(dim=-1).squeeze(dim=-1), p=2, dim=1) \
                    #             + F.normalize(attended_x1_global, p=2, dim=1)
                    #
                    # x2_global = F.normalize(x2_global.squeeze(dim=-1).squeeze(dim=-1), p=2, dim=1) \
                    #             + F.normalize(attended_x2_global, p=2, dim=1)

                    x1_global = attended_x1_global.reshape(N, C, -1).mean(axis=2)
                    if anch_pass_act:
                        anch_pass_act.append(attended_x1_global.detach().clone())

                    x2_global = attended_x2_global.reshape(N, C, -1).mean(axis=2)
                    if other_pass_act:
                        other_pass_act.append(attended_x2_global.detach().clone())

                if self.no_final_network:

                    x1_global = x1_global.view((x1_global.size()[0], -1))
                    x2_global = x2_global.view((x2_global.size()[0], -1))

                    pred = (x1_global * x2_global).sum(axis=1)

                    x1_global = F.normalize(x1_global, p=2, dim=1)
                    x2_global = F.normalize(x2_global, p=2, dim=1)
                    cos_sim = (x1_global * x2_global).sum(axis=1)

                    ret = (pred, cos_sim, x1_global, x2_global)
                else:
                    ret = self.sm_net(x1_global, x2_global, feats=feats, softmax=self.softmax)


                if self.loss == 'trpl_local':
                    pred, pdist, out1, out2 = ret
                    if attended_x1_global is not None:
                        ret = (pred, pdist, attended_x1_global, attended_x2_global)
                    else:
                        ret = (pred, pdist, x1_local[-1], x2_local[-1])

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

    def get_classifier_weights(self):
        if self.classifier:
            return self.classifier.weight
        elif self.local_features and self.local_features.classifier:
            return self.local_features.classifier[0].weight
        elif self.channel_attention and self.channel_attention.classifier:
            return self.channel_attention.classifier[0].weight
        else:
            return self.sm_net.get_classifier_weights()

    def get_sim_matrix(self, globals, locals, indices, bs=32):
        sim_matrix = np.zeros((len(locals), len(locals)), dtype=np.float32)
        min_dist_mask = np.ones((len(locals), len(locals)), dtype=bool)

        loader = DataLoader(dataset=Local_Feat_Dataset(locals=locals, globals=globals, indices=indices),
                            batch_size=bs,
                            num_workers=4,
                            pin_memory=True)


        with tqdm(total=len(loader), desc='PLEASE WORK!!') as t:
            for idx, batch in enumerate(loader):
                x1_local, x1_global, x2_local, x2_global, idx_pairs = batch

                res, _ = self.classify(globals=[x1_global.cuda(), x2_global.cuda()],
                                       locals=[[x1_local.cuda()], [x2_local.cuda()]],
                                       feats=False)

                sim_matrix[idx_pairs[:, 0], idx_pairs[:, 1]] = res.detach().cpu().numpy().flatten()
                min_dist_mask[idx_pairs[:, 0], idx_pairs[:, 1]] = False
                t.update()

        min_similarity = sim_matrix.min() - 1
        sim_matrix[min_dist_mask] = min_similarity

        return sim_matrix

def top_module(args, trained_feat_net=None, trained_sm_net=None, num_classes=1, mask=False, fourth_dim=False):
    if trained_sm_net is None:
        if args.loss == 'stopgrad':
            sm_net = StopGrad_MLP(args)
        else:
            sm_net = MLP(args)
    else:
        sm_net = trained_sm_net

    if args.no_final_network:
        sm_net = None

    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'vgg16': vgg16,
        'resnet101': resnet101,
        'deit16_224': deit16_224,
        'deit16_small_224': deit16_small_224,
    }

    use_pretrained = not (args.from_scratch)

    if trained_feat_net is None:
        print('Using pretrained model')
        if 'deit' not in args.feat_extractor:
            ft_net = model_dict[args.feat_extractor](args, pretrained=use_pretrained, num_classes=num_classes,
                                                     mask=mask,
                                                     fourth_dim=fourth_dim, output_dim=args.dim_reduction)
        else:
            ft_net = model_dict[args.feat_extractor]()
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

