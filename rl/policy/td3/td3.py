import numpy as np
import torch as th
import torch.nn as nn
from models import ActorImage, CriticImage, ActorState, CriticState
from os.path import join as pjoin
from .offpolicy import OffPolicyAgent
from .utils import polyak_update

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

def avg(l):
    return sum(l)/float(len(l))

class TD3(OffPolicyAgent):
    def __init__(self, *args, actor_lr:float=None, lr2:float=None, **kwargs):

        super().__init__(*args, **kwargs)

        self.lr2 = lr if lr2 is None else lr2
        self.actor_lr = lr if actor_lr is None else actor_lr

        self._create_models()
        self.to_save = ['critics', 'actor', 'opt_a', 'opt_c']

        print "TD3"

    def set_eval(self):
        self.actor.eval()
        for critic in self.critics:
            critic.eval()

    def set_train(self):
        self.actor.train()
        for critic in self.critics:
            critic.train()

    def _create_critic(self):
        if self.mode == 'image':
            n = CriticImage(action_dim=self.action_dim, img_stack=self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        elif self.mode == 'state':
            n = CriticState(state_dim=self.state_dim, action_dim=self.action_dim,
                    bn=self.bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + self.mode)
        return n

    def _create_actor(self):
        if self.mode == 'image':
            n = ActorImage(action_dim=self.action_dim, img_stack=self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        elif self.mode == 'state':
            n = ActorState(state_dim=self.state_dim, action_dim=self.action_dim,
                    bn=self.bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + self.mode)
        return n

    def _create_models(self):
        self.actor = self._create_actor()
        self.actor_t = self._create_actor()
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.actor, self.opt_a = self._create_opt(self.actor, self.actor_lr)

        self.critics = [self._create_critic() for _ in range(2)]
        self.critics_t = []
        for critic in self.critics:
            critic_t = self._create_critic()
            critic_t.load_state_dict(critic.state_dict())
            self.critics_t.append(critic_t)

        params = sum([list(c.parameters()) for c in self.critics])
        self.opt_c = self.create_opt(params)

    def select_action(self, state):
        # Copy as uint8
        state = self._process_state(state)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        state, state2, action, reward, done, extra_state, indices = \
            self._sample_to_dev(args.batch_size, beta=beta, num=num)

        with th.no_grad():
            # Add noise to action to add resilience
            noise = u.clone().data.normal_(0, args.policy_noise)
            noise = noise.clamp(-args.noise_clip, args.noise_clip)
            action2 = self.actor_t(state2) + noise

            # Compute the target Q value: min over all critic targets
            Qs_t = (c_t(state2, action2) for c_t in self.critics_t)
            Q_t = th.min(*Qs_t)
            Q_t = reward + done * args.discount * Q_t

        # Get current Q estimates
        Qs_now = (crit(state, action) for crit in self.critics)

        # Compute critic loss
        losses_c = ((Q_now - Q_t).pow(2) for Q_now, Q_t in zip(Qs_now, Qs_t))
        prios = ((sum(losses_c)/2.) + 1e-5).data.cpu().numpy()
        if weights is not None:
            losses_c = (loss * weights for loss in losses_c)
        loss_c = sum((loss_c.mean() for loss_c in losses_c))

        # Optimize the critics
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        replay_buffer.update_priorities(indices, prios)

        # Compute mean values for returning
        loss_c_mean = loss_c.item()
        Q_mean = avg((Q_now.mean().item() for Q_now in Qs_now))
        Q_max = max((Q_now.max().item() for Q_now in Qs_now))
        ret_loss_a = 0.

        # Policy updates
        if timesteps % args.policy_freq == 0:

            # Compute actor loss
            loss_a = -self.critics[0](state, self.actor(state)).mean()

            # Optimize the actor
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()

            for i in (0,1):
                polyak_update(self.critics[i].parameters(), self.critics_t[i].parameters(), args.tau)
            polyak_update(self.actor.parameters(), self.actor_t.parameters(), args.actor_tau)

        return loss_c_mean, loss_a.item(), Q_mean, Q_max
