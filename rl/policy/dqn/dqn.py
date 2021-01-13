import numpy as np
import torch as th
import os, sys, math
from os.path import join as pjoin

from rl.policy.common.offpolicy import OffPolicyAgent
from .policy import QState, QImage, QMixed, QImageAux, QMixedAux
from rl.policy.common.utils import polyak_update
import torch.cuda.amp as amp

device = th.device("cuda" if th.cuda.is_available() else "cpu")

# DQN

class DQN(OffPolicyAgent):
    def __init__(self, action_steps:int,
            tau=0.005, discount=0.99,
            cnn_net_arch=None, **kwargs):

        super().__init__(**kwargs)

        # odd dims should be centered at 0
        odd_dims = action_steps % 2 == 1
        # full range is 2: -1 to 1
        self.step_sizes = 2. / action_steps
        self.step_sizes[odd_dims] = 2. / (action_steps[odd_dims] - 1)
        self.total_steps = np.prod(action_steps)

        self.action_steps = action_steps
        self.discount = discount
        self.cnn_net_arch = cnn_net_arch
        self.to_save = ['q', 'q_t', 'q_opt']

        self._create_models()

        self.needs_quantization = True

        if self.amp:
            self.scaler = amp.GradScaler()

    def set_eval(self):
        self.q.eval()

    def set_train(self):
        self.q.train()

    def _create_model(self):
        if self.mode == 'image':
            if self.network == 'simple':
                if self.aux is None:
                    n = QImage(action_dim=self.total_steps, img_stack=self.total_stack,
                        bn=self.bn, img_dim=self.img_dim,
                        cnn_net_arch=self.cnn_net_arch,
                        drop=self.dropout).to(device)
                else:
                    if self.aux:
                        print("Aux")
                        aux_size = self.aux_size
                    else:
                        raise InvalidArgument()

                    n = QImage2Outs(action_dim=self.total_steps, img_stack=self.total_stack,
                        bn=self.bn, img_dim=self.img_dim, drop=self.dropout,
                        cnn_net_arch=self.cnn_net_arch,
                        aux_size=aux_size).to(device)

            elif self.network == 'densenet':
                n = QImageDenseNet(action_dim=self.total_steps,
                    img_stack=self.total_stack).to(device)
        elif self.mode == 'state':
            n = QState(state_dim=self.state_dim, action_dim=self.total_steps,
                    bn=self.bn, drop=self.dropout).to(device)
        elif self.mode == 'mixed':
            print("Mixed network")
            if self.network == 'simple':
                if not self.aux:
                    n = QMixed(state_dim=self.state_dim, action_dim=self.total_steps,
                        img_stack=self.total_stack, bn=self.bn,
                        cnn_net_arch=self.cnn_net_arch,
                        img_dim=self.img_dim, drop=self.dropout).to(device)
                else:
                    print("Aux")
                    aux_size = self.aux_size

                    n = QMixedAux(state_dim=self.state_dim, action_dim=self.total_steps,
                        img_stack=self.total_stack,
                        bn=self.bn, img_dim=self.img_dim, drop=self.dropout,
                        cnn_net_arch=self.cnn_net_arch,
                        aux_size=aux_size).to(device)

            elif self.network == 'densenet':
                n = QMixedDenseNet(action_dim=self.total_steps,
                    img_stack=self.total_stack, state_dim=self.state_dim).to(device)
        else:
            raise ValueError('Unrecognized mode ' + mode)

        if self.freeze:
            print("Freeze")
        return n

    def _create_models(self):
        self.q = self._create_model()
        self.q_t = self._create_model()

        self.q_t.load_state_dict(self.q.state_dict())
        self.q_opt = self._create_opt(self.q, self.lr)

        print("DDQN LR={}".format(self.lr))

    def _discrete_to_cont(self, discrete):
        ''' @discrete: (procs, 1) '''
        # If we have only the proc dimension, add a 1 to last dim
        if discrete.ndim < 2:
            discrete = np.expand_dims(discrete, -1)
        assert(discrete.ndim == 2)
        cont = []
        for dim_size, step_size in reversed(list(zip(
                self.action_steps, self.step_sizes))):
            num = (discrete % dim_size).astype(np.float32)
            num *= step_size
            num -= 1.
            cont = [np.transpose(num)] + cont
            discrete //= dim_size

        cont = np.concatenate(cont)
        cont = np.transpose(cont)
        return cont

    def _cont_to_discrete(self, cont):
        '''
        We turn an batch of ndarrays of continuous values to a batch of
        discrete values representing those continuous values.
        We leave them as floats here -- they'll be turned into longs later'''

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
        cont = np.array(cont)
        discrete = self._cont_to_discrete(cont)
        cont = self._discrete_to_cont(discrete)
        return cont

    def _copy_action_to_dev(self, u, batch_size):
        # u is the actions: still as ndarrays
        u = u.reshape((batch_size, self.action_dim))
        # Convert continuous action to discrete
        u = self._cont_to_discrete(u).reshape((batch_size, -1))
        u = th.LongTensor(u).to(device)
        return u

    def _get_q(self):
        return self.q

    def select_action(self, state):

        state = self._process_state(state)
        q = self._get_q()(state)

        if self.aux is not None:
            q = q[0]

        # Argmax along action dimension (not batch dim)
        max_action = th.argmax(q, -1).cpu().data.numpy()

        # Translate action choice to continous domain
        action = self._discrete_to_cont(max_action)
        return action

    def train(self, replay_buffer, batch_size, beta):

        # Sample replay buffer
        state, state2, action, reward, done, extra_state, indices = \
            self._sample_to_dev(replay_buffer, batch_size, beta=beta)

        Q_ts = self.q_t(state2)
        if self.aux is not None:
            Q_ts = Q_ts[0]
        Q_t = th.max(Q_ts, dim=-1, keepdim=True)[0]

        # Compute the target Q value
        # done: We use reverse of done to not consider future rewards

        Q_t = reward + (done * self.discount * Q_t).detach()

        compare_to = extra_state

        # Get current Q estimate
        with amp.autocast(enabled=self.amp):
            Q_now = self.q(state)

            if self.aux is not None:
                Q_now, predicted = Q_now

            Q_now = th.gather(Q_now, -1, action)

            # Compute Q loss
            q_loss = Q_now - Q_t
            q_loss = q_loss * q_loss
            prios = q_loss + 1e-5
            q_loss = q_loss.mean()

            if self.aux is not None:
                aux_loss = self.aux_loss(predicted, compare_to)

        prios = prios.data.cpu().numpy()

        if self.aux is not None:
            aux_losses.append(aux_loss.item())

            if self.freeze:
                self.q.freeze_some(False)

            self.q_opt.zero_grad()

            if self.amp:
                self.scaler.scale(aux_loss).backward()
                self.scaler.step(self.q_opt)
            else:
                aux_loss.backward()
                self.q_opt.step()

            if self.freeze:
                self.q.freeze_some(True) # Only backprop last layers

        # debug graph
        '''
        import thviz
        dot = thviz.make_dot(q_loss, params=dict(self.q.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

        # Optimize the model
        self.q_opt.zero_grad()

        if self.amp:
            self.scaler.scale(q_loss).backward()
            self.scaler.step(self.q_opt)
            self.scaler.update() # Only called after all steps in iter
        else:
            q_loss.backward()
            self.q_opt.step()

        replay_buffer.update_priorities(indices, prios)

        # Update the frozen target models
        polyak_update(self.q.parameters(), self.q_t.parameters(), self.tau)

        a_ret = None
        if self.aux is not None:
            a_ret = aux_loss.item()

        return q_loss.item(), a_ret, Q_now.mean().item(), Q_now.max().item()

    def test(self, replay_buffer, batch_size:int):
        [x, _, u, _, _, extra_state, _] = replay_buffer.sample(batch_size)
        length = len(u)

        state, extra_state = self._copy_sample_to_dev_small(x, extra_state, length)

        _, predict = self._get_q()(state)
        y = None
        if self.aux:
            y = extra_state.cpu().data.numpy()

        predict = predict.cpu().data.numpy()

        return y, predict

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)
        self.to_save = ['qs', 'qts', 'opts']

    def _create_models(self):
        # q is now 2 networks
        self.qs = [self._create_model() for _ in range(2)]
        self.qts = [self._create_model() for _ in range(2)]

        for q, qt in zip(self.qs, self.qts):
            qt.load_state_dict(q.state_dict())

        # debug
        #import pdb
        #pdb.set_trace()

        self.opts = [self._create_opt(q, self.lr) for q in self.qs]

        print("DDQN. LR={}".format(self.lr))

    def _get_q(self):
        return self.qs[0]

    def train(self, replay_buffer, batch_size, beta):

        q_losses, aux_losses, Q_max, Q_mean = [], [], [], []

        for num, (update_q, update_qt, opt) in \
                enumerate(zip(self.qs, self.qts, self.opts)):

            # Get samples
            state, state2, action, reward, done, extra_state, indices = \
                self._sample_to_dev(replay_buffer, batch_size, beta=beta, num=num)

            if self.aux is not None:
                if self.freeze:
                    update_q.freeze_some(False)

                compare_to = extra_state
                with amp.autocast(enabled=self.amp):
                        _, predicted = update_q(state)
                        aux_loss = self.aux_loss(predicted, compare_to)
                aux_losses.append(aux_loss.item())

                opt.zero_grad()

                if self.amp:
                    self.scaler.scale(aux_loss).backward()
                    self.scaler.step(opt)
                else:
                    aux_loss.backward()
                    opt.step()

                if self.freeze:
                    update_q.freeze_some(True) # Only backprop last layers

            # Not used for backprop
            with th.no_grad():
                Qt = [qt(state2) for qt in self.qts]
                if self.aux is not None:
                    Qt = [q[0] for q in Qt]
                Qt = th.min(*Qt)

                Qt_max, _ = th.max(Qt, dim=-1, keepdim=True)

                y = reward + (done * self.discount * Qt_max).detach()

            # Get current Q estimate
            with amp.autocast(enabled=self.amp):
                Q_now = update_q(state)

                if self.aux is not None:
                    Q_now, _ = Q_now

                Q_now = th.gather(Q_now, -1, action)

                # Compute loss
                q_loss = (Q_now - y)
                q_loss = q_loss * q_loss
                if replay_buffer.use_priorities:
                    prios = q_loss + 1e-5
                q_loss = q_loss.mean()

            if replay_buffer.use_priorities:
                prios = prios.data.cpu().numpy()
                replay_buffer.update_priorities(indices, prios)

            # Optimize the model
            opt.zero_grad()

            if self.amp:
                self.scaler.scale(q_loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
            else:
                q_loss.backward()
                opt.step()

            # Update the frozen target models
            polyak_update(update_q.parameters(), update_qt.parameters(), self.tau)

            q_losses.append(q_loss.item())
            Q_mean.append(Q_now.mean().item())
            Q_max.append(Q_now.max().item())

        aux_ret = None
        if self.aux is not None:
            aux_ret = np.mean(aux_losses)

        return np.mean(q_losses), aux_ret, np.mean(Q_mean), np.max(Q_max)

    def set_eval(self):
        for q in self.qs:
            q.eval()

    def set_train(self):
        for q in self.qs:
            q.train()
