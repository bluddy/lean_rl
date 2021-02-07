import numpy as np
import math
import torch.nn as nn
import torch as th
import torchvision.models as tmodels
import torch.nn.functional as F
from typing import List, Dict, Type, Tuple

feat_size = 7


def calc_features(img_stack):
    return img_stack

def make_linear(in_size:int, out_size:int, bn=False, drop=False, relu=True):
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

def create_mlp(start: int, net_arch: List[int], bn=True, drop=False, last=False, **kwargs) -> nn.Module:
    '''
    Create an MLP based on the net_arch
    :last: Should we add relu/bn/drop to last layer
    '''
    bn, drop, relu = bn, drop, True

    print(f"Linear: bn:{bn}, drop:{drop}")

    ll = []
    last_units = start
    for i, units in enumerate(net_arch):

        if last and i >= len(net_arch) - 1:
            bn, drop, relu = False, False, False

        ll.extend(make_linear(last_units, units, bn=bn, drop=drop, relu=relu))
        last_units = units

    x = nn.Sequential(*ll)
    x.out_size = net_arch[-1] if net_arch else start
    return x

class Dummy(nn.Module):
    '''
    Placeholder for input
    '''
    def __init__(self, size):
        super().__init__()

        self.out_size = size

    def forward(self, x:th.Tensor):
        return x

class CNN(nn.Module):
    def __init__(self, img_stack, drop=False,
            net_arch:Tuple[List[int], List[int], List[int]]='deep', img_dim:int=224, **kwargs):
        super().__init__()

        if net_arch is None or net_arch == 'deep':
            print('Deep ', end='')
            net_start_f = 8
            net_filters = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            net_strides = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2]
        elif net_arch == 'shallow':
            print('Shallow ', end='')
            net_start_f = 2
            net_filters = [3, 3, 3, 3, 3, 3, 3]
            net_strides = [2, 2, 2, 2, 2, 2, 2]
        else:
            net_start_f, net_filters, net_strides = net_arch

        ## input size:[img_stack, 224, 224]
        bn=True

        ll = []
        in_f = calc_features(img_stack)
        print(f"CNN. input_f:{in_f}, drop:{drop}, bn:{bn}")

        last_f = in_f
        f = net_start_f
        l = img_dim

        # Build up CNN
        for filter, stride in zip(net_filters, net_strides):
            assert(l > 1) # don't go too small

            if filter == 3:
                pad = 1
            elif filter == 5:
                pad = 2

            if stride == 2:
                l = int(math.ceil(l / 2))
                f *= 2
            elif stride == 4:
                l = int(math.ceil(l / 4))
                f *= 4

            ll.extend(make_conv(last_f, f, filter, stride, pad, bn=bn, drop=drop))
            last_f = f

        self.out_size = l * l * f

        ll.append(nn.Flatten())
        self.features = nn.Sequential(*ll)

    def forward(self, x):
        return self.features(x)

    def freeze(self, frozen):
        for p in self.features.parameters():
            p.requires_grad = not frozen


class Linear(nn.Module):
    def __init__(self, prev, net_arch:List[int], last=False, **kwargs):
        super().__init__()

        self.linear = create_mlp(prev.out_size, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        x = self.linear(x)
        return x

class MuxIn(nn.Module):
    '''
    Class to combine inputs from 2 incoming networks
    '''
    def __init__(self, net1: nn.Module, net2: nn.Module,
            net_arch: Tuple[List[int], List[int], List[int]], last=False, **kwargs):
        super().__init__()

        nets = [net1, net2]

        # Create linear layers if needed
        self.linears = []
        for arch, net in zip(net_arch[:-1], nets):
            self.linears.append(create_mlp(net.out_size, net_arch=arch, **kwargs))

        self.linear1 = self.linears[0]
        self.linear2 = self.linears[1]

        width = sum((x.out_size for x in self.linears))

        # Pass the last only if needed here
        self.linear_out = create_mlp(width, net_arch=net_arch[2], last=last, **kwargs)
        self.out_size = self.linear_out.out_size

    def freeze(self, idx:int, frozen:bool):
        for p in self.linears[idx].parameters():
            p.requires_grad = not frozen

    def forward(self, xs: Tuple[th.Tensor, th.Tensor]) -> th.Tensor:
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
        super().__init__()

        # Create input linear layer if needed
        self.linear_in = create_mlp(net.out_size, net_arch=net_arch[0], **kwargs)

        # Create output linear layers if needed
        self.linears = []
        for arch in net_arch[1:]:
            self.linears.append(create_mlp(self.linear_in.out_size, net_arch=arch, last=last, **kwargs))

        self.linear1 = self.linears[0]
        self.linear2 = self.linears[1]

        self.out_size = sum((x.out_size for x in self.linears))

    def freeze(self, idx:int, frozen:bool):
        for p in self.linears[idx].parameters():
            p.requires_grad = not frozen

    def forward(self, x: th.Tensor) -> Tuple[th.Tensor]:
        x = self.linear_in(x)
        xs = [linear(x) for linear in self.linears]
        return xs

