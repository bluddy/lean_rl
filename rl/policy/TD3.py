import numpy as np
import torch as th
import torch.nn as nn
from models import ActorImage, CriticImage, ActorState, CriticState
from os.path import join as pjoin
from .offpolicy import OffPolicyAgent

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

        print "TD3. LR={}, actor_LR={}, LR2={}".format(lr, actor_lr, lr2)

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

        self.critics, self.opts_c = list(zip(*[self._create_opt(c, self.lr)] for c in self.critics]))

    def select_action(self, state):
        # Copy as uint8
        state = self._process_state(state)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        state, state2, action, reward, done, extra_state, indices = \
            self._sample_to_dev(args.batch_size, beta=beta, num=num)

        # Add noise to action to add resilience
        noise = th.FloatTensor(u).data.normal_(0, args.policy_noise).to(device)
        noise = noise.clamp(-args.noise_clip, args.noise_clip)

        with th.no_grad():
            action2 = self.actor_t(state2) + noise

            # Compute the target Q value
            Qs_t = [c_t(state2, action2) for c_t in self.critics_t]
            Q_t = th.min(*Qs_t)
            Q_t = reward + (done * args.discount * Q_t)

        # Get current Q estimates
        Qs_now = [crit(state, action) for crit in self.critics]

        # Compute critic loss
        losses_c = [(Q_now - Q_t).pow(2) for Q_now, Q_t in zip(Qs_now, Qs_t)]
        prios = losses_c[0] + 1e-5
        prios = prios.data.cpu().numpy()
        if weights is not None:
            losses_c = [loss_c * weights for loss_c in losses_c]
        losses_c = [loss_c.mean() for loss_c in losses_c]

        # Optimize the critics
        for opt_c, loss_c in zip(self.opts_c, losses_c):
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        replay_buffer.update_priorities(indices, prios)

        # Compute mean values for returning
        loss_c_mean = avg([loss_c.item() for loss_c in losses_c])
        Q_mean = avg([Q_now.mean().item() for Q_now in Qs_now])
        Q_max = max([Q_now.max().item() for Q_now in Qs_now])
        ret_loss_a = 0.

        # Policy updates
        if timesteps % args.policy_freq == 0:

            # Compute actor loss
            loss_a = -self.critics[0](state, self.actor(state)).mean()

            # Optimize the actor
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()

            # Update the frozen target models
            for c, c_t in zip(self.critics, self.critics_t):
                for p, p_t in zip(c.parameters(), c_t.parameters()):
                    p_t.data.copy_(args.tau * p.data + (1 - args.tau) * p_t.data)

            tau = args.actor_tau
            for p, p_t in zip(self.actor.parameters(), self.actor_t.parameters()):
                p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)

            ret_loss_a = loss_a.item()

        return loss_c_mean, ret_loss_a, Q_mean, Q_max
