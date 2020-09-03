import torch as th
from torch import nn as nn

class Actor(object):
    """
    Actor network (policy) for SAC.

    :param use_sde: (bool) Whether to use State Dependent Exploration or not
    """
    def __init__(self, use_sde=False):

        self.use_sde = use_sde
        self.latent_pi_net = pass # TODO
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(last_layer_dim, action_dim)
        self.log_std = nn.Linear(last
