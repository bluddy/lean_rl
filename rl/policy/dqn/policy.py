import numpy as np
import torch as th
import torch.nn as nn
from rl.policy.common.models import *

class QState(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = [100, 50, action_dim]

        self.linear = create_mlp(state_dim, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        return self.linear(x)

class QImage(nn.Module):
    def __init__(self, action_dim, fc_net_arch=None, cnn_net_arch=None, **kwargs):
        super().__init__()

        if fc_net_arch is None:
            fc_net_arch = [100, 50, action_dim]

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.linear = Linear(self.cnn, net_arch=fc_net_arch, last=True, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        return x

class QMixed(nn.Module):
    def __init__(self, state_dim, action_dim, cnn_net_arch, mux_net_arch=None, **kwargs):
        super().__init__()

        if mux_net_arch is None:
            mux_net_arch = ([50, 20], [50, 20], [action_dim])

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.dummy = Dummy(state_dim)
        self.mux_in = MuxIn(self.cnn, self.dummy, net_arch=mux_net_arch, last=True, **kwargs)

    def forward(self, x):
        y = self.cnn(x[0])
        z = self.dummy(x[1])
        x = self.mux_in((y, z))
        return x

class QImageAux(nn.Module):
    def __init__(self, action_dim, aux_size, mux_net_arch=None, cnn_net_arch=None, **kwargs):
        super().__init__()

        if mux_net_arch is None:
            mux_net_arch = ([20], [100, 50, action_dim], [100, 50, aux_size])

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.mux_out = MuxOut(self.cnn, net_arch=mux_net_arch, last=True, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mux_out(x)
        return x

class QMixedAux(nn.Module):
    ''' QMixed with auxiliary output coming out of the features
        Use state shape of QState
    '''
    def __init__(self, state_dim, action_dim, aux_size,
            cnn_net_arch=None, net_arch1=None, net_arch2=None, **kwargs):
        super().__init__()

        print("QMixedAux")

        if net_arch1 is None:
            net_arch1 = ([50, 20], [50, 20], [100, 50])
        if net_arch2 is None:
            net_arch2 = ([], [action_dim], [aux_size])

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.state = Dummy(state_dim)
        self.mux_in = MuxIn(self.cnn, self.state, net_arch=net_arch1, **kwargs)
        self.mux_out = MuxOut(self.mux_in, net_arch=net_arch2, last=True, **kwargs)

    def freeze_some(self, frozen):
        self.mux_in.freeze(idx=0, frozen=frozen)
        self.cnn.freeze(frozen=frozen)

    def forward(self, x):
        y = self.cnn(x[0])
        z = self.state(x[1])
        x = self.mux_in((y,z))
        x = self.mux_out(x)
        return x

