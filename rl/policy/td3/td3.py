import numpy as np
import torch as th
import copy
from os.path import join as pjoin

from rl.policy.common.offpolicy import OffPolicyAgent
from rl.policy.common.actor_critic import ActorImage, CriticImage, ActorState, CriticState
from rl.policy.common.utils import polyak_update
import torch.cuda.amp as amp

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# One actor, 2 critics

def avg(l):
    return sum(l)/float(len(l))

class TD3(OffPolicyAgent):
    def __init__(self, policy_noise, noise_clip, policy_freq, cnn_net_arch=None, actor_lr:float=None, **kwargs):

        super().__init__(**kwargs)

        self.actor_lr = self.lr if actor_lr is None else actor_lr
        self.cnn_net_arch = cnn_net_arch
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.train_step = 0

        self.last_loss_a_np = 0

        self.to_save = ['critics', 'critics_t', 'actor', 'actor_t', 'opt_a', 'opt_c']
        self._create_models()

        if self.amp:
            self.scaler = amp.GradScaler()
        print("TD3")

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
                    bn=self.bn, img_dim=self.img_dim, drop=self.dropout,
                    cnn_net_arch=self.cnn_net_arch).to(device)
        elif self.mode == 'state':
            n = CriticState(state_dim=self.state_dim, action_dim=self.action_dim,
                    bn=self.bn, drop=self.dropout).to(device)
        else:
            raise ValueError('Unrecognized mode ' + self.mode)
        return n

    def _create_actor(self):
        if self.mode == 'image':
            n = ActorImage(action_dim=self.action_dim, img_stack=self.total_stack,
                    bn=self.bn, img_dim=self.img_dim, drop=self.dropout,
                    cnn_net_arch=self.cnn_net_arch).to(device)
        elif self.mode == 'state':
            n = ActorState(state_dim=self.state_dim, action_dim=self.action_dim,
                    bn=self.bn, drop=self.dropout).to(device)
        else:
            raise ValueError('Unrecognized mode ' + self.mode)
        return n

    def _create_models(self):
        self.actor = self._create_actor()
        self.actor_t = copy.deepcopy(self.actor)

        self.opt_a = self._create_opt(self.actor, self.actor_lr)

        self.critics =   [self._create_critic() for _ in range(2)]
        self.critics_t = [copy.deepcopy(c) for c in self.critics]

        params = (par for m in self.critics for par in m.parameters())
        self.opt_c = self._create_opt(params, self.lr)

    def select_action(self, state):
        # Copy as uint8
        state = self._process_state(state)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, batch_size, discount, tau, beta):

        # Sample replay buffer
        state, state2, action, reward, not_done, extra_state, indices = \
            self._sample_to_dev(replay_buffer, batch_size, beta=beta)

        with th.no_grad():
            # Add noise to action to add resilience
            noise = th.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            action2 = self.actor_t(state2) + noise
            action2 = action2.clamp(-1., 1.) # due to noise

            # Compute the target Q value: min over all critic targets
            Qs_t = (c_t(state2, action2) for c_t in self.critics_t)
            Q_t = th.min(*Qs_t)
            Q_t = reward + not_done * discount * Q_t

        # Get current Q estimates
        with amp.autocast(enabled=self.amp):
            Qs_now = [crit(state, action) for crit in self.critics]

            # Compute critic loss
            loss_c = sum(((Q_now - Q_t).pow(2) for Q_now in Qs_now))
            if replay_buffer.use_priorities:
                prios = (loss_c + 1e-5).data.cpu().numpy()
                replay_buffer.update_priorities(indices, prios)
            loss_c = loss_c.mean()

        self.opt_c.zero_grad()

        # Optimize the critics
        if self.amp:
            self.scaler.scale(loss_c).backward()
            self.scaler.step(self.opt_c)
        else:
            loss_c.backward()
            self.opt_c.step()

        # Policy updates
        if not self.train_step % self.policy_freq:

            # Compute actor loss
            with amp.autocast(enabled=self.amp):
                loss_a = -self.critics[0](state, self.actor(state)).mean()

            self.opt_a.zero_grad()

            # Optimize the actor
            if self.amp:
                self.scaler.scale(loss_a).backward()
                self.scaler.step(self.opt_a)
            else:
                loss_a.backward()
                self.opt_a.step()

            for c, c_t in (self.critics, self.critics_t):
                polyak_update(c.parameters(), c_t.parameters(), tau)
            polyak_update(self.actor.parameters(), self.actor_t.parameters(), tau)

            self.last_loss_a_np = loss_a.item()

        # Don't forget to update scaler
        if self.amp:
            self.scaler.update()

        # Compute mean values for returning
        with th.no_grad():
            Q_mean = avg(list(Q_now.mean().item() for Q_now in Qs_now))
            Q_max = max((Q_now.max().item() for Q_now in Qs_now))

        self.train_step += 1

        return loss_c.item(), self.last_loss_a_np, Q_mean, Q_max
