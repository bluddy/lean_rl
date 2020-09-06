import numpy as np
import torch.nn as nn
import torch as th
import torchvision.models as tmodels
import torch.nn.functional as F
from typing import List, Dict, Type, Tuple

feat_size = 7


def calc_features(img_stack):
    return img_stack

def make_linear(in_size, out_size, bn=False, drop=False, relu=True):
    l = []
    l.append(nn.Linear(in_size, out_size))
    if relu:
        l.append(nn.ReLU(inplace=True))
    if bn:
        l.append(nn.BatchNorm1d(out_size))
    if drop:
        l.append(nn.Dropout(p=0.1))
    return l

def make_conv(in_channels, out_channels, kernel_size, stride, padding, bn=False, drop=False):
    l = []
    l.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    l.append(nn.ReLU(inplace=True))
    l.append(nn.BatchNorm2d(out_channels))
    if drop:
        l.append(nn.Dropout2d(0.2))
    return l

def create_mlp(start: int, net_arch: List[int], bn=True, drop=False, last=False) -> nn.Module:
    '''
    Create an MLP based on the net_arch
    :last: Should we add relu/bn/drop to last layer
    '''
    bn, drop, relu = bn, drop, True

    ll = []
    last_units = start
    for i, units in enumerate(net_arch):

        if last and i == len(net_arch) - 1:
            bn, drop, relu = False, False, False

        ll.extend(make_linear(last_units, units, bn=bn, drop=drop, relu=relu))
        last_unit = units

    x = nn.Sequential(*ll)
    x.out_size = net_arch[-1] if net_arch else start
    return x

class Dummy(nn.Module):
    '''
    Placeholder for input
    '''
    def __init__(self):
        super().__init__(size)

        self.out_size = size

    def forward(self, x:th.Tensor):
        return x

class CNN(nn.Module):
    def __init__(self, img_stack, drop=False, net_arch=(8,[],[]), img_dim=224, **kwargs):
        super().__init__()

        if net_arch[1] == []:
            net_start_f = 8
            net_filters = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            net_strides = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2]
        else:
            net_start_f, net_filters, net_strides = net_arch

        ## input size:[img_stack, 224, 224]
        print("BaseImage. drop:{}".format(drop))
        bn=True

        ll = []
        in_f = calc_features(img_stack)

        last_f = in_f
        f = net_start_f
        l = img_dim

        # Build up CNN
        for filter, stride in zip(net_filters, net_strides):
            if filter == 3:
                pad = 1
            elif filter == 5:
                pad = 2

            last_f = f
            if stride == 2:
                f *= 2
                l /= 2
            elif stride == 4:
                f *= 4
                l /= 4

            ll.extend(make_conv(last_f, f, filter, stride, pad, bn=bn, drop=drop))

        self.out_size = l * l * f

        ll.append(nn.Flatten())
        self.features = nn.Sequential(*ll)

class MuxIn(nn.Module):
    '''
    Class to combine inputs from 2 incoming networks
    '''
    def __init__(self, net1: nn.Module, net2: nn.Module,
            net_arch: Tuple[List[int], List[int], List[int]), last=False, **kwargs)
        super().__init()

        self.nets = [net1, net2]
        # Create linear layers if needed
        self.linears = []
        for arch, net in zip(net_arch[:-1], self.nets):
            self.linears.append(create_mlp(net.out_size, net_arch=arch, **kwargs))

        width = sum((x.out_size for x in self.linears))

        # Pass the last only if needed here
        self.linear_out = create_mlp(width, net_arch=net_arch[2], last=last, **kwargs)
        self.out_size = self.linear_out.out_size

    def freeze(self, idx:int, frozen:bool):
        for p in self.linears[idx].parameters():
            p.requires_grad = not frozen

    def forward(self, xs: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        xs = [net(x) for x, net in zip(xs, self.nets)]
        xs = [linear(x) for x, linear in zip(xs, self.linears)]
        x = th.cat(xs, dim=1)
        x = self.linear_out(x)
        return x

class MuxOut(nn.Module):
    '''
    Class to split output to 2 networks 
    '''
    def __init__(self, net: nn.Module,
            net_arch: Tuple[List[int], List[int], List[int]], last=False, **kwargs):
        super().__init()

        self.net = net

        # Create input linear layer if needed
        self.linear_in = create_mlp(net.out_size, net_arch=arch, **kwargs)

        # Create output linear layers if needed
        self.linears = []
        for arch in net_arch[1:]:
            self.linears.append(create_mlp(width, net_arch=arch, last=last, **kwargs))

        self.out_size = sum((x.out_size for x in self.linears))

    def freeze(self, idx:int, frozen:bool):
        for p in self.linears[idx].parameters():
            p.requires_grad = not frozen

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor]:
        x = self.net(x)
        x = self.linear_in(x)
        xs = [linear(x) for linear in self.linears]
        return xs

'''
class ActorImage(BaseImage):
    def __init__(self, action_dim, bn=False, **kwargs):
        super(ActorImage, self).__init__(bn=bn, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        self.linear = nn.Sequential(*ll)

        self.out_angular = nn.Linear(100, action_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        x = self.out_angular(x)
        #x = th.clamp(x, min=-1., max=1.)
        x = th.tanh(x)
        return x

class CriticImage(BaseImage):
    def __init__(self, action_dim, bn=False, **kwargs):
        super(CriticImage, self).__init__(bn=bn, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 100, bn=bn))
        ll.extend(make_linear(100, 1, bn=False, relu=False))
        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = self.features(x)
        x = th.cat([x, u], 1)
        x = self.linear(x)
        return x

class QImage(BaseImage):
    def __init__(self, action_dim, bn=True, drop=False, **kwargs):
        super(QImage, self).__init__(bn=bn, drop=drop, **kwargs)

        ll = []
        #ll.extend(make_linear(self.latent_dim, action_dim, bn=bn, drop=drop))
        ll.extend(make_linear(self.latent_dim, action_dim, bn=False, drop=False, relu=False))
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x

class QImage2Outs(BaseImage):
    ''' QImage with two outputs coming out of the featres
        Use state shape of QState
    '''
    def __init__(self, action_dim, aux_size, drop=False, reduced_dim=10, **kwargs):
        super(QImage2Outs, self).__init__(drop=drop, **kwargs)

        print("QImage2Outs: reduced_dim={}, drop={}".format(reduced_dim, drop))

        bn=True
        d = reduced_dim

        # Map features to small state space
        ll = []
        ll.extend(make_linear(self.latent_dim, d, bn=bn, drop=drop))
        ll.extend(make_linear(d, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        self.linear = nn.Sequential(*ll)

        # RL part
        ll = []
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))
        self.linear1 = nn.Sequential(*ll)

        # Aux part
        ll = []
        ll.extend(make_linear(50, aux_size, bn=False, drop=False, relu=False))
        self.linear2 = nn.Sequential(*ll)

    def freeze_some(self, frozen):
        raise ValueError("Freezing not supported in QImage2Outs")

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        y = self.linear1(x)
        z = self.linear2(x)
        return y,z

class QImageDenseNet(nn.Module):
    def __init__(self, action_dim, img_stack, pretrained=False):
        super(QImageDenseNet, self).__init__()

        print("QImageDenseNet. action_dim:{}, img_stack:{}".format(action_dim, img_stack))

        model = tmodels.densenet121(pretrained=False)
        layers = list(model.features.children())
        if img_stack != 3:
            layers[0] = nn.Conv2d(img_stack, 64, 7, stride=2, padding=3, bias=False)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(1024, action_dim)

    def forward(self, x):
        #import pdb
        #pdb.set_trace()

        x = F.relu(self.features(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x

class QMixedDenseNet(QImageDenseNet):
    def __init__(self, action_dim, state_dim, img_stack):
        self.latent_dim = 100
        super(QMixedDenseNet, self).__init__(action_dim=self.latent_dim, img_stack=img_stack)

        print("QMixedDenseNet. action_dim:{}, img_stack:{}, state_dim:{}".format(action_dim, img_stack, state_dim))

        bn = True
        drop = False

        ll = []
        ll.extend(make_linear(self.latent_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 100, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 100, bn=bn, drop=drop))
        self.linear2 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(200, action_dim, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def forward(self, x):
        img, state = x
        x = self.model(img)
        x = self.linear1(x)
        y = self.linear2(state)
        z = self.linear3(th.cat((x, y), dim=-1))
        return z

class ActorState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=False):
        super(ActorState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend(make_linear(100, action_dim, bn=False, drop=False, relu=False))

        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.linear(x)
        #x = th.clamp(x, min=-1., max=1.)
        x = th.tanh(x)
        return x


class CriticState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=False):
        super(CriticState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim + action_dim, 400, bn=bn))
        ll.extend(make_linear(400, 300, bn=bn))
        ll.extend(make_linear(300, 100, bn=bn))
        ll.extend(make_linear(100, 1, bn=False, drop=False, relu=False))

        self.linear = nn.Sequential(*ll)

    def forward(self, x, u):
        x = th.cat([x, u], 1)
        x = self.linear(x)
        return x

class QState(nn.Module):
    def __init__(self, state_dim, action_dim, bn=True, drop=False, **kwargs):
        super(QState, self).__init__()

        ll = []
        ll.extend(make_linear(state_dim, 100, bn=bn, drop=drop))
        ll.extend(make_linear(100, 50, bn=bn, drop=drop))
        ll.extend(make_linear(50, action_dim, drop=False, bn=False, relu=False))

        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.linear(x)
        return x


class QMixed2(nn.Module):
    def __init__(self, img_stack, bn=True, drop=False, img_dim=224, deep=False):
        super(QMixed2, self).__init__()

        ## input size:[img_stack, 224, 224]
        print("QMixed2. drop:{}, deep:{}, bn:{}".format(drop, deep, bn))

        ll = []
        in_f = calc_features(img_stack)
        #[4, (3, 2), (3, 1), (3, 2), (3, 1), (3, 2), (3, 1), (3, 2), (3, 1), (3, 2), (3, 1)]
        if img_dim == 224:
            d = 4; l = img_dim
            ll.extend(make_conv(in_f, d,  1, 1, 1, bn=bn, drop=drop)) # flatten colors, 224
            d2 = 8; l = l / 2
            ll.extend(make_conv(d, d2, 3, 2, 1, bn=bn, drop=drop)) # 112
            if deep:
                ll.extend(make_conv(d2, d2, 3, 1, 1, bn=bn, drop=drop))
            d = 16; l = l / 2
            ll.extend(make_conv(d2, d,  3, 2, 1, bn=bn, drop=drop)) # 56
            if deep:
                ll.extend(make_conv(d, d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 32; l = l / 2
            ll.extend(make_conv(d,  d2,  3, 2, 1, bn=bn, drop=drop)) # 28
            if deep:
                ll.extend(make_conv(d2, d2,  3, 1, 1, bn=bn, drop=drop))
            d = 64; l = l / 2
            ll.extend(make_conv(d2,  d,  3, 2, 1, bn=bn, drop=drop)) # 14
            if deep:
                ll.extend(make_conv(d, d,  3, 1, 1, bn=bn, drop=drop))
            d2 = 128; l = l / 2
            ll.extend(make_conv(d, d2,  3, 2, 1, bn=bn, drop=drop)) # 7
            if deep:
                ll.extend(make_conv(d2, d2,  3, 1, 1, bn=bn, drop=drop))
            d = d2
            self.latent_dim = l * l * d
        else:
            raise ValueError(str(img_dim) + " is not a valid img_dim")

        ll.extend([nn.Flatten()])
        self.features = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, 100, bn=bn, drop=drop))
        self.linear1 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(state_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, 100, bn=bn, drop=drop))
        self.linear2 = nn.Sequential(*ll)

        ll = []
        ll.extend(make_linear(200, action_dim, bn=False, drop=False, relu=False))
        self.linear3 = nn.Sequential(*ll)

    def forward(self, x):
        img, state = x
        x = self.features(img)
        x = self.linear1(x)
        y = self.linear2(state)
        x = self.linear3(th.cat((x, y), dim=-1))
        return x

class QImageSoftMax(BaseImage):
    ''' Image network with softmax '''
    def __init__(self, action_dim, bn=True, drop=False, **kwargs):
        super(QImageSoftMax, self).__init__(bn=bn, drop=drop, **kwargs)

        ll = []
        ll.extend(make_linear(self.latent_dim, 400, bn=bn, drop=drop))
        ll.extend(make_linear(400, action_dim, bn=False, drop=False, relu=False))
        ll.extend([nn.LogSoftmax(dim=-1)])
        self.linear = nn.Sequential(*ll)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x
'''
