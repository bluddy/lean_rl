import numpy as np
import torch
import torch.nn as nn
import os, sys, math
import torch.nn.functional as F
from os.path import join as pjoin
from models import QState, QImage, QMixed, QImageSoftMax, QImageDenseNet, QMixedDenseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN

# We have greyscale, and then one RGB

class DQN(object):
    def __init__(self, state_dim, action_dim, action_steps, stack_size,
            mode, network, lr=1e-4, img_depth=3, bn=True, img_dim=224,
            load_encoder='', amp=False, use_orig_q=False,
            deep=False, dropout=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_steps = action_steps
        self.deep = deep
        self.dropout = dropout

        # odd dims should be centered at 0
        odd_dims = action_steps % 2 == 1
        # full range is 2: -1 to 1
        self.step_sizes = 2. / action_steps
        self.step_sizes[odd_dims] = 2. / (action_steps[odd_dims] - 1)
        self.total_steps = np.prod(action_steps)

        self.total_stack = stack_size * img_depth
        self.mode = mode
        self.network = network
        self.lr = lr
        self.bn = bn
        self.img_dim = img_dim
        self.load_encoder = load_encoder
        self.amp = amp
        self.use_orig_q = use_orig_q

        self._create_models()

        if self.amp:
            import amp

        print "LR=", lr

    def set_eval(self):
        self.q.eval()

    def set_train(self):
        self.q.train()

    def _create_model(self):
        if self.mode == 'image':
            if self.network == 'simple':
                n = QImage(action_dim=self.total_steps, img_stack=self.total_stack,
                    bn=self.bn, img_dim=self.img_dim, deep=self.deep,
                    drop=self.dropout).to(device)
            elif self.network == 'densenet':
                n = QImageDenseNet(action_dim=self.total_steps,
                    img_stack=self.total_stack).to(device)
        elif self.mode == 'state':
            n = QState(state_dim=self.state_dim, action_dim=self.total_steps,
                    bn=self.bn, deep=self.deep, drop=self.dropout).to(device)
        elif self.mode == 'mixed':
            if self.network == 'simple':
                n = QMixed(state_dim=self.state_dim, action_dim=self.total_steps,
                    img_stack=self.total_stack, bn=self.bn,
                    img_dim=self.img_dim, deep=self.deep, drop=self.dropout).to(device)
            elif self.network == 'densenet':
                n = QMixedDenseNet(action_dim=self.total_steps,
                    img_stack=self.total_stack, state_dim=self.state_dim).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def _create_models(self):
        self.q = self._create_model()
        self.q_target = self._create_model()

        self.q_target.load_state_dict(self.q.state_dict())
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        if self.load_encoder != '':
            print "Loading encoder model..."
            for model in [self.q, self.q_target]:
                    model.encoder.load_state_dict(torch.load(self.load_encoder))

        if self.amp:
            self.q, self.q_optimizer = amp.initialize(
                    self.q, self.q_optimizer, opt_level='O1')

    def _discrete_to_cont(self, discrete):
        ''' @discrete: (procs, 1) '''
        # If we have only the proc dimension, add a 1 to last dim
        if discrete.ndim < 2:
            discrete = np.expand_dims(discrete, -1)
        assert(discrete.ndim == 2)
        cont = []
        for dim_size, step_size in reversed(zip(
                self.action_steps, self.step_sizes)):
            num = (discrete % dim_size).astype(np.float32)
            num *= step_size
            num -= 1.
            cont = [np.transpose(num)] + cont
            discrete /= dim_size

        cont = np.concatenate(cont)
        cont = np.transpose(cont)
        #print "XXX cont_after.shape", cont.shape
        return cont

    def _cont_to_discrete(self, cont):
        '''
        We turn an batch of ndarrays of continuous values to a batch of
        discrete values representing those continuous values.
        We leave them as floats here -- they'll be turned into longs later'''

        #print "cont =", cont # debug
        #print "shape =", cont.shape, "step_size =", self.step_sizes, "action_steps", self.action_steps # debug

        # Continuous dimensions: (batch, action_dim)
        assert(cont.ndim == 2)
        total = np.zeros((cont.shape[0],), dtype=np.int64)
        cont = np.transpose(cont)

        for v, dim_size, step_size in zip(
                cont, self.action_steps, self.step_sizes):
            v += 1.             # Start range at 0
            v /= step_size      # Quantize
            v[v < 0] = 0        # Bound 0 < v < dim_size - 1
            v[v >= dim_size] = dim_size - 1
            v = v.astype(np.int64)
            total *= dim_size   # Shift total by 1
            total += v # Add to total

        return np.expand_dims(total, -1) # Add dim for action

    def quantize_continuous(self, cont):
        ''' We need to quantize the actions to the allowable values
            so we don't have the noise process produce something we can't
            reproduce
        '''
        cont2 = np.array(cont)
        discrete = self._cont_to_discrete(cont2)
        cont3 = self._discrete_to_cont(discrete)
        return cont3

    def _process_state(self, state):
        # Copy as uint8
        if self.mode == 'image':
            if state.ndim < 4:
                state = np.expand_dims(state, 0)
            state = torch.from_numpy(state).to(device).float() # possibly missing squeeze(1)
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
        # u is the actions: still as ndarrays
        u = u.reshape((batch_size, self.action_dim))
        # Convert continuous action to discrete
        u = self._cont_to_discrete(u).reshape((batch_size, -1))
        u = torch.LongTensor(u).to(device)
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

        state = self._process_state(state)
        q = self.q(state)

        #print "XXX q.shape: ", q.shape
        #print "XXX DQN q: ", q

        # Argmax along action dimension (not batch dim)
        max_action = torch.argmax(q, -1).cpu().data.numpy()

        #print "XXX DQN max_action:", max_action
        #print "XXX max_action shape: ", max_action.shape

        # Translate action choice to continous domain
        action = self._discrete_to_cont(max_action)
        #print "XXX DQN action:", action
        #print "XXX DQN action.shape:", action.shape
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        [x, y, u, r, d, qorig, indices, w], qorig_prob = replay_buffer.sample(
            args.batch_size, beta=beta)

        length = len(u)

        state, state2, action, reward, done, qorig, weights = \
            self._copy_sample_to_dev(x, y, u, r, d, qorig, w, length)

        Q_ts = self.q_target(state2)
        Q_t = torch.max(Q_ts, dim=-1, keepdim=True)[0]

        # Compute the target Q value
        # done: We use reverse of done to not consider future rewards

        Q_t = reward + (done * args.discount * Q_t).detach()

        # Overwrite with q_orig
        if self.use_orig_q and qorig_prob > 0.:
            choose_probs = torch.rand(length).to(device)
            mask = choose_probs.le(qorig_prob)
            indices = mask.nonzero().squeeze(1)
            Q_t[indices] = qorig[indices]

        # Get current Q estimate
        Q_now = self.q(state)
        Q_now = torch.gather(Q_now, -1, action)

        # Compute loss
        q_loss = (Q_now - Q_t).pow(2)
        prios = q_loss + 1e-5
        prios = prios.data.cpu().numpy()
        if weights is not None:
            q_loss *= weights
        q_loss = q_loss.mean()

        # debug graph
        '''
        import torchviz
        dot = torchviz.make_dot(q_loss, params=dict(self.q.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

        #print("weights = ", w, "prios = ", prios)

        # Optimize the model
        self.q_optimizer.zero_grad()

        if self.amp:
            with amp.scale_loss(q_loss, self.q_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            q_loss.backward()

        if args.clip_grad is not None:
            nn.utils.clip_grad_value_(self.q.parameters(), args.clip_grad)

        self.q_optimizer.step()

        replay_buffer.update_priorities(indices, prios)

        # Update the frozen target models
        for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
            p_t.data.copy_(args.tau * p.data + (1 - args.tau) * p_t.data)

        return q_loss.item(), None, Q_now.mean().item(), Q_now.max().item()

    def save(self, path):
        torch.save(self.q.state_dict(), os.path.join(path, 'q.pth'))
        torch.save(self.q_target.state_dict(), os.path.join(path, 'q_target.pth'))

    def load(self, path):
        self.q.load_state_dict(torch.load(os.path.join(path, 'q.pth')))
        self.q_target.load_state_dict(torch.load(os.path.join(path, 'q_target.pth')))

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)

    def _create_models(self):
        # q is now 2 networks
        self.qs = [self._create_model() for _ in range(2)]
        self.qts = [self._create_model() for _ in range(2)]

        for q, qt in zip(self.qs, self.qts):
            qt.load_state_dict(q.state_dict())

        self.opts = [torch.optim.Adam(q.parameters(), lr=self.lr) for q in self.qs]

    def select_action(self, state):

        state = self._process_state(state)
        q = self.qs[0](state)

        max_action = torch.argmax(q, -1).cpu().data.numpy()

        action = self._discrete_to_cont(max_action)
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        losses, Q_max, Q_mean = [], [], []

        for num, (update_q, update_qt, opt) in \
                enumerate(zip(self.qs, self.qts, self.opts)):

            # Get samples
            [x, y, u, r, d, qorig, indices, w], qorig_prob = \
                replay_buffer.sample(args.batch_size, beta=beta, num=num)
            length = len(u)

            state, state2, action, reward, done, qorig, weights = \
                self._copy_sample_to_dev(x, y, u, r, d, qorig, w, length)

            #from rl.utils import ForkablePdb
            #ForkablePdb().set_trace()

            Qt = [qt(state2) for qt in self.qts]
            Qt = torch.min(*Qt)

            Qt_max, _ = torch.max(Qt, dim=-1, keepdim=True)

            y = reward + (done * args.discount * Qt_max).detach()

            # Get current Q estimate
            Q_now = update_q(state)
            Q_now = torch.gather(Q_now, -1, action)

            # Compute loss
            loss = (Q_now - y).pow(2)
            prios = loss + 1e-5
            prios = prios.data.cpu().numpy()
            if weights is not None:
                loss *= weights
            loss = loss.mean()

            # Optimize the model
            opt.zero_grad()
            loss.backward()

            if args.clip_grad is not None:
                nn.utils.clip_grad_norm_(update_q.parameters(), args.clip_grad)

            opt.step()

            replay_buffer.update_priorities(indices, prios)

            # Update the frozen target models
            for p, pt in zip(update_q.parameters(), update_qt.parameters()):
                pt.data.copy_(args.tau * p.data + (1 - args.tau) * pt.data)

            losses.append(loss.item())
            Q_mean.append(Q_now.mean().item())
            Q_max.append(Q_now.max().item())

        return np.mean(losses), None, np.mean(Q_mean), np.max(Q_max)

    def set_eval(self):
        for q in self.qs:
            q.eval()

    def set_train(self):
        for q in self.qs:
            q.train()

    def save(self, path):
        for i, (q, qt) in enumerate(zip(self.qs, self.qts)):
            torch.save(q.state_dict(), pjoin(path, 'q{}.pth'.format(i)))
            torch.save(qt.state_dict(), pjoin(path, 'qt{}.pth'.format(i)))

    def load(self, path):
        for i, (q, qt) in enumerate(zip(self.qs, self.qts)):
            q.load_state_dict(torch.load(pjoin(path, 'q{}.pth'.format(i))))
            qt.load_state_dict(torch.load(pjoin(path, 'qt{}.pth'.format(i))))

class BatchDQN(DQN):
    def __init__(self, n_samples=100, **kwargs):
        super(BatchDQN, self).__init__(**kwargs)

        self.action_weights = self._create_weight_model()
        self.opt_w = torch.optim.Adam(self.action_weights.parameters(), lr=self.lr)
        self.loss_w = nn.NLLLoss()
        self.n_samples = n_samples

        self.qt = self.q_target
        self.opt_q = self.q_optimizer

    def _create_weight_model(self):
        if self.mode == 'image':
            n = QImageSoftMax(self.total_steps, self.total_stack,
                    bn=self.bn, img_dim=self.img_dim).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return n

    def select_action(self, state):
        #import pdb
        #pdb.set_trace()

        state = self._process_state(state)
        qv = self.q(state)

        a_weights = torch.exp(self.action_weights(state))
        sample_idx = torch.multinomial(a_weights, self.n_samples, replacement=True)
        sampled_q = torch.gather(qv, -1, sample_idx)

        max_q_idx = torch.argmax(sampled_q, -1, keepdim=True)
        max_action = torch.gather(sample_idx, -1, max_q_idx).cpu().data.numpy()

        action = self._discrete_to_cont(max_action)
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        losses, Q_max, Q_mean = [], [], []

        # Get samples
        [x, y, u, r, d, qorig, indices, w], qorig_prob = \
            replay_buffer.sample(args.batch_size, beta=beta)
        length = len(u)

        state, state2, action, reward, done, qorig, weights = \
            self._copy_sample_to_dev(x, y, u, r, d, qorig, w, length)

        #import pdb
        #pdb.set_trace()

        with torch.no_grad():
            a_weights = torch.exp(self.action_weights(state2))
            sample_idx = torch.multinomial(a_weights, self.n_samples, replacement=True)
            Qt = self.qt(state2)
            Qt = torch.gather(Qt, -1, sample_idx)
            Qt_max, _ = torch.max(Qt, dim=-1, keepdim=True)
            y = reward + (done * args.discount * Qt_max)

        # Get current Q estimate
        Q_now = self.q(state)
        Q_now = torch.gather(Q_now, -1, action)

        q_loss = (Q_now - y).pow(2)
        prios = q_loss + 1e-5
        prios = prios.data.cpu().numpy()
        if weights is not None:
            q_loss *= weights
        q_loss = q_loss.mean()

        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        replay_buffer.update_priorities(indices, prios)


        # --- Update the action weight network ---
        a_logps = self.action_weights(state)
        w_loss = self.loss_w(a_logps, action.squeeze(-1))
        self.opt_w.zero_grad()
        w_loss.backward()
        self.opt_w.step()
        # ---

        # Update the frozen target models
        for p, pt in zip(self.q.parameters(), self.qt.parameters()):
            pt.data.copy_(args.tau * p.data + (1 - args.tau) * pt.data)

        return q_loss.item(), q_loss.item(), Q_now.mean().item(), Q_now.max().item()

    def set_eval(self):
        super(BatchDQN, self).set_eval()
        self.action_weights.eval()

    def set_train(self):
        super(BatchDQN, self).set_train()
        self.action_weights.train()

    def save(self, path):
        super(BatchDQN, self).save(path)
        torch.save(self.action_weights.state_dict(), pjoin(path, 'aw.pth'))

    def load(self, path):
        super(BatchDQN, self).load(path)
        self.action_weights.load_state_dict(torch.load(pjoin(path, 'aw.pth')))
