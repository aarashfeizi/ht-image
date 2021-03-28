import torch

from models.MLP import *
from models.resnet import *
from models.vgg import *

class AttentionBlock(nn.Module):

    def __init__(self, input_channels):
        super(AttentionBlock, self).__init__()
        self.att = nn.Sequential(nn.Conv2d(in_channels=input_channels , out_channels=input_channels, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.att(x).squeeze(dim=2).squeeze(dim=2)
        return x

class AttentionModule(nn.Module):

    def  __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.att_blocks = []
        self.fc_blocks = []
        total_channels = 0
        for c in in_channels:
            self.att_blocks.append(AttentionBlock(c))
            self.fc_blocks.append(nn.Sequential(nn.Linear(in_features=2 * c, out_features=c), nn.ReLU()))
            total_channels += c # todo residual connection

        self.att_concat = nn.Sequential(nn.Linear(in_features=total_channels, out_features=2048), nn.ReLU())

    def forward(self, x1s, x2s):
        rets = []
        for x1, x2, a, fc in zip(x1s, x2s, self.att_blocks, self.fc_blocks):
            x1 = a(x1)
            x2 = a(x2)
            x = fc(utils.vector_merge_function(x1, x2, method='concat'))
            rets.append(x)

        ret = self.att_concat(torch.cat(rets, dim=1))

        return ret

class TopModel(nn.Module):

    def __init__(self, ft_net, sm_net, aug_mask=False, attention=False):
        super(TopModel, self).__init__()
        self.ft_net = ft_net
        self.sm_net = sm_net
        self.aug_mask = aug_mask
        self.attention = attention

        if self.attention:
            self.attention_mod = AttentionModule([#256,
                                              # 512,
                                              1024,
                                              2048]) # only for resnet50
        else:
            self.attention_mod = None

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

            if self.attention:
                att_vec = self.attention_mod(x1_all[:-1], x2_all[:-1])
                # ret =
            else:
                ret = self.sm_net(x1_f, x2_f, feats=feats)

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
        ft_net = model_dict[args.feat_extractor](args, pretrained=use_pretrained, num_classes=num_classes, mask=mask, fourth_dim=fourth_dim)
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

    return TopModel(ft_net=ft_net, sm_net=sm_net, aug_mask=(mask and fourth_dim), attention=args.attention)
