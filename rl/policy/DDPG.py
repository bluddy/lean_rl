import numpy as np
import torch
import torch.nn as nn
import os, sys
import torch.nn.functional as F
from models import ActorImage, CriticImage, ActorState, CriticState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


''' Utilities '''

# NOTE: Batchnorm is a problem for these algorithms. We need consistency
# and determinism, especially for the actor. Batchnorm seems to break that.

# We have greyscale, and then one RGB

class DDPG(object):
    def __init__(self, state_dim, action_dim, img_stack,
            max_action, mode, lr, bn=False, actor_lr=None):

        self.max_action = max_action
        self.action_dim = action_dim
        self.mode = mode
        actor_lr = lr if actor_lr is None else actor_lr
        if self.mode == 'image':
            self.actor = ActorImage(action_dim, img_stack, max_action).to(device)
            self.actor_target = ActorImage(action_dim, img_stack, max_action).to(device)
            self.critic = CriticImage( action_dim, img_stack).to(device)
            self.critic_target = CriticImage( action_dim, img_stack).to(device)
        elif self.mode == 'state':
            self.actor = ActorState(state_dim, action_dim, max_action, bn=bn).to(device)
            self.actor_target = ActorState(state_dim, action_dim, max_action, bn=bn).to(device)
            self.critic = CriticState(state_dim, action_dim, bn=bn).to(device)
            self.critic_target = CriticState(state_dim, action_dim, bn=bn).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=actor_lr)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=lr)

    def select_action(self, state):
        # Copy as uint8
        if self.mode == 'image':
            state = torch.from_numpy(state).unsqueeze(0).to(device).float()
            state /= 255.0
        elif self.mode == 'state':
            state = torch.from_numpy(state).to(device).float()
            # print("state size: " + str(state.size()))
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return self.actor(state).cpu().data.numpy().flatten()

    def copy_sample_to_device(self, x, y, u, r, d, w, batch_size):
        # Copy as uint8
        x = torch.from_numpy(x).squeeze(1).to(device).float()
        y = torch.from_numpy(y).squeeze(1).to(device).float()
        if self.mode == 'image':
            x /= 255.0 # Normalize
            y /= 255.0 # Normalize
        u = u.reshape((batch_size, self.action_dim))
        u = torch.FloatTensor(u).to(device)
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        w = w.reshape((batch_size, -1))
        w = torch.FloatTensor(w).to(device)
        return x, y, u, r, d, w

    def train(self, replay_buffer, timesteps, beta_PER, args):

        batch_size = args.batch_size
        discount = args.discount
        tau = args.tau
        actor_tau = args.actor_tau

        # Sample replay buffer
        x, y, u, r, d, indices, w = replay_buffer.sample(
                batch_size, beta=beta_PER)

        state, next_state, action, reward, done, weights = \
                self.copy_sample_to_device(x, y, u, r, d, w, batch_size)

        next_action = self.actor_target(next_state)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + (done * discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = (current_Q - target_Q).pow(2)
        prios = critic_loss + 1e-5
        critic_loss *= weights
        prios = prios.data.cpu().numpy()
        critic_loss = critic_loss.mean()

        # debug graph
        '''
        import torchviz
        dot = torchviz.make_dot(critic_loss, params=dict(self.critic.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

        #print("weights = ", w, "prios = ", prios)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        replay_buffer.update_priorities(indices, prios)

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(actor_tau * param.data + (1 - actor_tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, path):
        torch.save(self.actor.state_dict(), os.path.join(path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pth'))
        torch.save(self.actor_target.state_dict(), os.path.join(path, 'actor_target.pth'))
        torch.save(self.critic_target.state_dict(), os.path.join(path, 'critic_target.pth'))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'critic.pth')))
        self.actor_target.load_state_dict(torch.load(os.path.join(path, 'actor_target.pth')))
        self.critic_target.load_state_dict(torch.load(os.path.join(path, 'critic_target.pth')))

