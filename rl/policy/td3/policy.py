import numpy as np
import torch as th
import torch.nn as nn
import os, sys, math

device = th.device("cuda" if th.cuda.is_available() else "cpu")

try:
    import apex.amp as amp
except ImportError:
    pass

class ActorImage(nn.Module):
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

class CriticImage(nn.Module):
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

