import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ActorImage, CriticImage, ActorState, CriticState
from os.path import join as pjoin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

# We have greyscale, and then one RGB

class DDPG(object):
    def __init__(self, state_dim, action_dim, stack_size,
            mode, lr=1e-4, img_depth=3, bn=True, actor_lr=None,
            img_dim=224):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.total_stack = stack_size * img_depth
        self.mode = mode

        self.lr = lr
        self.bn = bn
        self.img_dim = img_dim

        self.actor_lr = lr if actor_lr is None else actor_lr

        self._create_models()

        print "LR={}, actor LR={}".format(lr, actor_lr)

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()

    def _create_critic(self):
        if self.mode == 'image':
            n = CriticImage(self.action_dim, self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        elif self.mode == 'state':
            n = CriticState(self.state_dim, self.action_dim, bn=self.bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def _create_actor(self):
        if self.mode == 'image':
            n = ActorImage(self.action_dim, self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        elif self.mode == 'state':
            n = ActorState(self.state_dim, self.action_dim, max_action=1.0,
                    bn=self.bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def _create_models(self):
        self.actor = self._create_actor()
        self.actor_t = self._create_actor()
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = self._create_critic()
        self.critic_t = self._create_critic()
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

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

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        [x, y, u, r, d, qorig, indices, w], qorig_prob = \
                replay_buffer.sample(args.batch_size, beta=beta)

        state, state2, action, reward, done, qorig, weights = \
            self._copy_sample_to_dev(x, y, u, r, d, qorig, w, len(u))

        action2 = self.actor_t(state2)

        # Compute the target Q value
        Qt = self.critic_t(state2, action2)
        Qt = reward + (done * args.discount * Qt).detach()

        # Get current Q estimate
        Q_now = self.critic(state, action)

        # Compute critic loss
        loss_c = (Q_now - Qt).pow(2)
        prios = loss_c + 1e-5
        prios = prios.data.cpu().numpy()
        if weights is not None:
            loss_c *= weights
        loss_c = loss_c.mean()

        # Optimize the critic
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        replay_buffer.update_priorities(indices, prios)

        # Compute actor loss
        loss_a = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        # Update the frozen target models
        for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
            pt.data.copy_(args.tau * p.data + (1 - args.tau) * pt.data)

        for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
            pt.data.copy_(args.actor_tau * p.data + (1 - args.actor_tau) * pt.data)

        return loss_c.item(), loss_a.item(), Q_now.mean().item(), Q_now.max().item()

    def save(self, path):
        torch.save(self.actor.state_dict(), pjoin(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), pjoin(path, 'critic.pth'))
        torch.save(self.actor_t.state_dict(), pjoin(path, 'actor_t.pth'))
        torch.save(self.critic_t.state_dict(), pjoin(path, 'critic_t.pth'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(pjoin(path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(pjoin(path, 'critic.pth')))
        self.actor_t.load_state_dict(torch.load(pjoin(path, 'actor_t.pth')))
        self.critic_t.load_state_dict(torch.load(pjoin(path, 'critic_t.pth')))

