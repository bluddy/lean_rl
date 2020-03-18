import numpy as np
import torch
from os.path import join as pjoin
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy
from models import ActorImage, CriticImage, ActorState, CriticState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3:
    def __init__(self, state_dim, action_dim, img_stack,
            max_action, mode, lr, actor_lr=None, lr2=None,
            bn=False, img_dim=224, load_encoder=''):

        self.action_dim = action_dim
        self.max_action = max_action
        self.mode = mode
        lr2 = lr if lr2 is None else lr2
        actor_lr = lr if actor_lr is None else actor_lr

        if self.mode == 'image':
            def create_actor():
                return ActorImage(action_dim, img_stack, max_action,
                        bn=bn, img_dim=img_dim).to(device)

            self.actor = create_actor()
            self.actor_target = create_actor()

            def create_critic():
                return CriticImage(action_dim, img_stack, bn=bn,
                         img_dim=img_dim).to(device)

            self.critics = [create_critic() for _ in xrange(2)]
            self.critic_targets = [create_critic() for _ in xrange(2)]

            # Load encoder if requested
            if load_encoder != '':
                print "Loading encoder model..."
                for model in [self.actor] + self.critics:
                     model.encoder.load_state_dict(torch.load(load_encoder))

        elif self.mode == 'state':
            self.actor = ActorState(
                    state_dim, action_dim, max_action, bn=bn).to(device)
            self.actor_target = ActorState(
                    state_dim, action_dim, max_action, bn=bn).to(device)
            def create_critic():
                return CriticState(state_dim, action_dim, bn=bn).to(device)

            self.critics = [create_critic() for _ in xrange(2)]
            self.critic_targets = [create_critic() for _ in xrange(2)]
        else:
            raise ValueError('Unrecognized mode ' + mode)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
               lr=actor_lr)
        #self.actor_optimizer = torch.optim.SGD(self.actor.parameters(),
        #        lr=actor_lr, momentum=0.)

        self.critic_optimizers = []
        for critic, critic_target, crit_lr in zip(
                self.critics, self.critic_targets, (lr,lr2)):
            critic_target.load_state_dict(critic.state_dict())
            self.critic_optimizers.append(torch.optim.Adam(
                critic.parameters(), lr=crit_lr))
            #self.critic_optimizers.append(torch.optim.SGD(
            #    critic.parameters(), lr=crit_lr, momentum=0.))

    def select_action(self, state):
        # Copy as uint8
        if self.mode == 'image':
            state = torch.from_numpy(state).unsqueeze(0).to(device).float()
            state /= 255.0
        elif self.mode == 'state':
            state = torch.from_numpy(state).to(device).float()
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

        # Noise to add smoothing
        policy_noise = args.policy_noise
        noise_clip = args.noise_clip
        policy_freq = args.policy_freq

        # Sample replay buffer
        x, y, u, r, d, indices, w = replay_buffer.sample(
                batch_size, beta=beta_PER)

        state, next_state, action, reward, done, weights = \
                self.copy_sample_to_device(x, y, u, r, d, w, batch_size)

        # Select action according to policy and add clipped noise
        # for smoothing
        noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)

        # NOTE: May need to scale noise for max_action
        next_action = self.actor_target(next_state) + noise
        next_action = torch.clamp(next_action,
                -self.max_action,self.max_action)

        # Compute the target Q value
        target_Qs = [c_t(next_state, next_action)
                for c_t in self.critic_targets]
        target_Q = torch.min(*target_Qs)
        target_Q = reward + (done * discount * target_Q).detach()

        # Get current Q estimates
        current_Qs = [crit(state, action) for crit in self.critics]

        # Compute critic loss
        critic_losses = []
        critic_mean_losses = []
        for current_Q in current_Qs:
            critic_losses.append(weights * (current_Q - target_Q).pow(2))
        for critic_loss in critic_losses:
            critic_mean_losses.append(critic_loss.mean())

        # No good way to do priorities for 2 networks
        prios = (critic_losses[0] + critic_losses[1]) / 2. + 1e-5

        # Optimize the critics
        for c_opt, crit_loss in zip(self.critic_optimizers, critic_mean_losses):
            c_opt.zero_grad()
            crit_loss.backward()
            c_opt.step()
#             print("indices len: " + str(len(indices)))
#             print("prios size:" + str(prios.size()))

        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        # Delayed policy updates
        ret_actor_loss = 0.
        if timesteps % policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critics[0](state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for critic, critic_t in zip(self.critics, self.critic_targets):
                for param, param_t in zip(critic.parameters(),
                        critic_t.parameters()):
                    param_t.data.copy_(tau * param.data +
                            (1 - tau) * param_t.data)

            for param, param_t in zip(self.actor.parameters(),
                    self.actor_target.parameters()):
                param_t.data.copy_(actor_tau * param.data + (1 - actor_tau) * param_t.data)

            ret_actor_loss = actor_loss.item()

        mean_crit_loss = sum([c.item() for c in critic_mean_losses]) / 2.
        return mean_crit_loss, ret_actor_loss

    def save(self, path):
        torch.save(self.actor.state_dict(), pjoin(path, 'actor.pth'))
        torch.save(self.actor_target.state_dict(), pjoin(path, 'actor_t.pth'))
        for i, (critic, critic_t) in enumerate(
                zip(self.critics, self.critic_targets)):
            torch.save(critic.state_dict(), pjoin(path, 'critic{}.pth'.format(i)))
            torch.save(critic_t.state_dict(), pjoin(path,
                'critic_t{}.pth'.format(i)))

    def load(self, path):
        self.actor.load_state_dict(torch.load(pjoin(path, 'actor.pth')))
        self.actor_target.load_state_dict(torch.load(pjoin(path, 'actor_t.pth')))
        for i, (critic, critic_t) in enumerate(
                zip(self.critics, self.critic_targets)):
            critic.load_state_dic(torch.load(
                pjoin(path, 'critic{}.pth'.format(i))))
            critic_t.load_state_dic(torch.load(
                pjoin(path, 'critic_t{}.pth'.format(i))))
