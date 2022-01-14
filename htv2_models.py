import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models import ResNet as tResNet

from models.resnet import conv1x1, conv3x3, BasicBlock, Bottleneck

import models.pooling as pooling
import utils

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class ResNet(tResNet):

    def __init__(self, block, layers, num_classes, four_dim=False, pooling_method='spoc', output_dim=0,
                 layer_norm=False):
        super(ResNet, self).__init__(block, layers)
        self.gradients = None
        self.activations = None

        print(f'Pooling method: {pooling_method}')
        if pooling_method == 'spoc':
            self.pool = self.avgpool
        elif pooling_method == 'gem':
            self.pool = pooling.GeM()
        elif pooling_method == 'mac':
            self.pool = pooling.MAC()
        elif pooling_method == 'rmac':
            self.pool = pooling.RMAC()
        else:
            raise Exception(f'Pooling method {pooling_method} not implemented... :(')

        previous_output = self.layer4[-1].conv3.out_channels if type(self.layer4[-1]) == Bottleneck else self.layer4[
            -1].conv2.out_channels

        if layer_norm:
            self.layer_norm = nn.LayerNorm(previous_output, elementwise_affine=False)
        else:
            self.layer_norm = None

        if output_dim != 0:
            self.last_conv = nn.Conv2d(in_channels=previous_output, out_channels=output_dim,
                                       kernel_size=(1, 1), stride=(1, 1))
        else:
            self.last_conv = None

        self.rest = nn.Sequential(self.conv1,
                                  self.bn1,
                                  self.relu,
                                  self.maxpool,
                                  self.layer1,
                                  self.layer2,
                                  self.layer3,
                                  self.layer4)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.modules()):
            module.eval()
            module.train = lambda _: None

    def activations_hook(self, grad):
        self.gradients = grad.clone()

    def forward(self, x, is_feat=False, hook=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f0 = x

        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.last_conv is not None:
            x = self.last_conv(x)  # downsampling channels for dim reduction

        if hook:
            x.register_hook(self.activations_hook)
            self.activations = x.clone()

        f4 = x
        x = self.pool(x)

        feat = x
        x = torch.flatten(x, 1)

        if is_feat:
            return feat, [f1, f2, f3, f4]
        else:
            return x

    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self):
        return self.activations

    def load_my_state_dict(self, state_dict, four_dim):

        own_state = self.state_dict()
        for name, param in state_dict.items():

            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            if four_dim and name == 'conv1.weight':
                print('Augmented zero initialized!!!')
                zeros = torch.zeros(size=[64, 1, 7, 7])
                param = torch.cat((param, zeros), axis=1)

            own_state[name].copy_(param)


def _resnet(arch, block, layers, pretrained, progress, num_classes, pooling_method='spoc',
            mask=False, fourth_dim=False, project_path='.', output_dim=0, pretrained_model='',
            layer_norm=False, **kwargs):
    model = ResNet(block, layers, num_classes, four_dim=(mask and fourth_dim),
                   pooling_method=pooling_method, output_dim=output_dim,
                   layer_norm=layer_norm, **kwargs)


    return model

def resnet50(args, pretrained=False, progress=True, num_classes=1, mask=False, fourth_dim=False, output_dim=0,
             **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes,
                   project_path='',
                   mask=mask, fourth_dim=fourth_dim, pooling_method='spoc',
                   output_dim=args.sz_embedding, layer_norm=False, **kwargs)
    # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, num_classes, project_path=args.get('project_path'),
    #                mask=mask, fourth_dim=fourth_dim, pooling_method='spoc', output_dim=output_dim, pretrained_model=args.pretrained_model, **kwargs)


class TopModule(nn.Module):
    def __init__(self, args, encoder):
        super(TopModule, self).__init__()
        self.metric = args.metric
        self.encoder = encoder
        self.logits_net = None
        self.temeperature = 1

        if self.metric == 'mlp':
            self.logits_net = nn.Sequential(nn.Linear(in_features=2 * args.sz_embedding,
                                                      out_features=args.sz_embedding),
                                            nn.ReLU(),
                                            nn.Linear(in_features=args.sz_embedding,
                                                      out_features=1))

    def get_preds(self, embeddings):
        if self.metric == 'cosine':
            norm_embeddings = F.normalize(embeddings, p=2)
            sims = torch.matmul(norm_embeddings, norm_embeddings.T)
            preds = (sims + 1) / 2  # maps (-1, 1) to (0, 1)

            preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'euclidean':
            euclidean_dist = utils.pairwise_distance(embeddings)

            euclidean_dist = euclidean_dist / self.temperature

            preds = 2 * nn.functional.sigmoid(-euclidean_dist)  # maps (0, +inf) to (1, 0)
            sims = -euclidean_dist
            # preds = torch.clamp(preds, min=0.0, max=1.0)
        elif self.metric == 'mlp':
            bs = embeddings.shape[0]
            indices = torch.tensor([[i, j] for i in range(bs) for j in range(bs)]).flatten()
            logits = self.logits_net(embeddings[indices].reshape(bs * bs, -1))

            sims = logits / self.temperature
            preds = nn.functional.sigmoid(sims)
        else:
            raise Exception(f'{self.metric} not supported in Top Module')

        return preds, sims

    def forward(self, imgs):
        embeddings = self.encoder(imgs)
        # preds, sims = self.get_preds(embeddings)

        return embeddings


def get_top_module(args):
    encoder = resnet50(args)
    return TopModule(args, encoder)
