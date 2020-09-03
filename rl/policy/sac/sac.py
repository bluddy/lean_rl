import numpy as np
import torch as th
import torch.nn as nn
from models import ActorImage, CriticImage, ActorState, CriticState
from os.path import join as pjoin
from .offpolicy import OffPolicyAgent
from .utils import polyak_update
from .policy import Actor

device = th.device("cuda" if th.cuda.is_available() else "cpu")




class SAC(OffPolicyAgent):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, use_sde = False, clip_mean= 2.0, **kwargs)

        self.target_entropy = target_entropy
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.clip_mean = clip_mean
        self.use_sde = use_sde # TODO

        self._create_models()
        self.to_save = {}

        print "SAC"

    def _create_models(self):
        pass

    def train(self, replay_buffer, timesteps, batch_size, discount, tau, beta):
        pass
