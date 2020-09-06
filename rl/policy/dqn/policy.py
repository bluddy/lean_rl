import numpy as np
import torch as th
import torch.nn as nn
import os, sys, math

device = th.device("cuda" if th.cuda.is_available() else "cpu")

try:
    import apex.amp as amp
except ImportError:
    pass

class QState(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = [100, 50, action_dim]

        self.linear = create_mlp(state_dim, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        return self.linear(x)

class QImage(nn.Module):
    def __init__(self, action_dim, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = [100, 50, action_dim]

        cnn = CNN(**kwargs)
        self.model = Linear(cnn, net_arch, last=True, **kwargs)

    def forward(self, x):
        return self.model(x)

class QMixed(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = ([50, 20], [50, 20], [action_dim])

        cnn = CNN(**kwargs)
        dummy = Dummy(state_dim)
        self.model = MuxIn(cnn, dummy, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        return self.model(x)

class QImageAux(nn.Module):
    def __init__(self, action_dim, aux_size, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = ([20], [100, 50, action_dim], [100, 50, aux_size])

        cnn = CNN(**kwargs)
        self.mux_out = MuxOut(cnn, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        return self.mux_out(x)


class QMixedAux(nn.Module):
    ''' QMixed with auxiliary output coming out of the features
        Use state shape of QState
    '''
    def __init__(self, state_dim, action_dim, aux_size, net_arch1=None, net_arch2=None, **kwargs):
        super().__init__()

        print("QMixedAux")

        if net_arch1 is None:
            net_arch1 = ([50, 20], [50, 20], [100, 50])
        if net_arch2 is None:
            net_arch2 = ([], [action_dim], [aux_size])

        self.cnn = CNN(**kwargs)
        state = Dummy(state_dim)
        self.mux_in = MuxIn(self.cnn, state, net_arch=net_arch1, **kwargs)
        self.mux_out = MuxOut(self.mux_in, net_arch=net_arch2, last=True, **kwargs)

    def freeze_some(self, frozen):
        self.mux_in.freeze(first=True, frozen=frozen)
        self.cnn.freeze(frozen=frozen)

    def forward(self, x):
        return self.mux_out(x)

