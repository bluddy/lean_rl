import numpy as np
import torch as th
import torch.nn as nn
from rl.policy.common.models import *

class ActorState(nn.Module):
    ''' Actor can only represent actions between -1 and 1 on each dimension '''
    def __init__(self, state_dim, action_dim, net_arch=None, **kwargs):
        super().__init__()

        if net_arch is None:
            net_arch = [100, 50, action_dim]

        self.linear = create_mlp(state_dim, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x):
        x = self.linear(x)
        #x = th.clamp(x, min=-1., max=1.)
        x = th.tanh(x)
        return x

class CriticState(nn.Module):
    def __init__(self, state_dim, action_dim, net_arch=None, **kwargs):
        super(CriticState, self).__init__()

        if net_arch is None:
            net_arch = [state_dim + action_dim, 100, 50, 1]

        self.linear = create_mlp(state_dim + action_dim, net_arch=net_arch, last=True, **kwargs)

    def forward(self, x, u):
        x = th.cat([x, u], 1)
        x = self.linear(x)
        return x

class ActorImage(nn.Module):
    def __init__(self, action_dim, fc_net_arch=None, cnn_net_arch=None, **kwargs):
        super().__init__()

        if fc_net_arch is None:
            fc_net_arch = [100, 50, action_dim]

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.linear = Linear(self.cnn, net_arch=fc_net_arch, last=True, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        x = self.linear(x)
        #x = th.clamp(x, min=-1., max=1.)
        x = th.tanh(x)
        return x

class CriticImage(nn.Module):
    def __init__(self, action_dim, fc_net_arch=None, cnn_net_arch=None, **kwargs):
        super().__init__()

        if fc_net_arch is None:
            fc_net_arch = ([], [], [100, 50, 20, 1])

        self.cnn = CNN(net_arch=cnn_net_arch, **kwargs)
        self.actions = Dummy(action_dim)
        self.mux = MuxIn(self.cnn, self.actions, net_arch=fc_net_arch, last=True, **kwargs)

    def forward(self, x, u):
        x = self.cnn(x)
        y = self.actions(u)
        x = self.mux((x,y))
        return x

