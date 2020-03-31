import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ActorImage, CriticImage, ActorState, CriticState
from os.path import join as pjoin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

def avg(l):
    return sum(l)/float(len(l))

class TD3(object):
    def __init__(self, state_dim, action_dim, stack_size,
            mode, lr=1e-4, img_depth=3, bn=True, actor_lr=None, lr2=None,
            img_dim=224):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.total_stack = stack_size * img_depth
        self.mode = mode

        self.lr = lr
        self.bn = bn
        self.img_dim = img_dim

        self.lr2 = lr if lr2 is None else lr2
        self.actor_lr = lr if actor_lr is None else actor_lr

        self._create_models()

        print "LR={}, actor_LR={}, LR2={}".format(lr, actor_lr, lr2)

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
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def _create_actor(self):
        if self.mode == 'image':
            n = ActorImage(action_dim=self.action_dim, img_stack=self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        elif self.mode == 'state':
            n = ActorState(state_dim=self.state_dim, action_dim=self.action_dim,
                    bn=self.bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def _create_models(self):
        self.actor = self._create_actor()
        self.actor_t = self._create_actor()
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critics = [self._create_critic() for _ in range(2)]
        self.critics_t = [self._create_critic() for _ in range(2)]
        for critic, critic_t in zip(self.critics, self.critics_t):
            critic_t.load_state_dict(critic.state_dict())

        self.opts_c = []
        for critic in self.critics:
            self.opts_c.append(torch.optim.Adam(critic.parameters(), lr=self.lr))

    def _process_state(self, state):
        # Copy as uint8
        if self.mode == 'image':
            if state.ndim < 4:
                state = np.expand_dims(state, 0)
            state = torch.from_numpy(state).to(device).float()
            state /= 255.0
        elif self.mode == 'state':
            state = torch.from_numpy(state).to(device).float()
        elif self.mode == 'mixed':
            img = state[0]
            if img.ndim < 4:
                img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).to(device).float()
            img /= 255.0
            state2 = torch.from_numpy(state[1]).to(device).float()
            state = (img, state2)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return state

    def _copy_sample_to_dev(self, x, y, u, r, d, qorig, w, batch_size):
        x = self._process_state(x)
        y = self._process_state(y)
        u = torch.FloatTensor(u).to(device) # actions
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        if qorig is not None:
            qorig = qorig.reshape((batch_size, -1))
            qorig = torch.FloatTensor(qorig).to(device)
        if w is not None:
            w = w.reshape((batch_size, -1))
            w = torch.FloatTensor(w).to(device)
        return x, y, u, r, d, qorig, w

    def select_action(self, state):
        # Copy as uint8
        state = self._process_state(state)
        action = self.actor(state).cpu().data.numpy()
        return action

    def _copy_sample_to_dev(self, x, y, u, r, d, qorig, w, batch_size):
        x = self._process_state(x)
        y = self._process_state(y)
        u = torch.FloatTensor(u).to(device) # actions
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        if qorig is not None:
            qorig = qorig.reshape((batch_size, -1))
            qorig = torch.FloatTensor(qorig).to(device)
        if w is not None:
            w = w.reshape((batch_size, -1))
            w = torch.FloatTensor(w).to(device)
        return x, y, u, r, d, qorig, w

    def select_action(self, state):
        # Copy as uint8
        state = self._process_state(state)
        action = self.actor(state).cpu().data.numpy()
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        [x, y, u, r, d, qorig, indices, w], qorig_prob = \
                replay_buffer.sample(args.batch_size, beta=beta)

        state, state2, action, reward, done, qorig, weights = \
                self._copy_sample_to_dev(x, y, u, r, d, qorig, w, len(u))

        # Add noise to action to add resilience
        noise = torch.FloatTensor(u).data.normal_(0, args.policy_noise).to(device)
        noise = noise.clamp(-args.noise_clip, args.noise_clip)

        with torch.no_grad():
            action2 = self.actor_t(state2) + noise

            # Compute the target Q value
            Qs_t = [c_t(state2, action2) for c_t in self.critics_t]
            Q_t = torch.min(*Qs_t)
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

    def save(self, path):
        torch.save(self.actor.state_dict(), pjoin(path, 'actor.pth'))
        torch.save(self.actor_t.state_dict(), pjoin(path, 'actor_t.pth'))
        for i, (critic, critic_t) in enumerate(zip(self.critics, self.critics_t)):
            torch.save(critic.state_dict(),
                pjoin(path, 'critic{}.pth'.format(i)))
            torch.save(critic_t.state_dict(),
                pjoin(path, 'critic{}_t.pth'.format(i)))

    def load(self, path):
        self.actor.load_state_dict(torch.load(pjoin(path, 'actor.pth')))
        self.actor_t.load_state_dict(torch.load(pjoin(path, 'actor_t.pth')))
        for i, (critic, critic_t) in enumerate(zip(self.critics, self.critics_t)):
            critic.load_state_dic(torch.load(
                pjoin(path, 'critic{}.pth'.format(i))))
            critic_t.load_state_dic(torch.load(
                pjoin(path, 'critic{}_t.pth'.format(i))))
