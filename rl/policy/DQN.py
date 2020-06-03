import numpy as np
import torch
import torch.nn as nn
import os, sys, math
import torch.nn.functional as F
from os.path import join as pjoin
from models import QState, QImage, QMixed, QImageSoftMax, QImageDenseNet, QMixedDenseNet, QImage2Outs
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN

e24 = pow(2,24)

class DQN(object):
    def __init__(self, state_dim, action_dim, action_steps, stack_size,
            mode, network, lr=1e-4, img_depth=3, img_dim=224,
            amp=False, dropout=False, aux=None, aux_size=6, reduced_dim=10,
            depthmap_mode=False):
        '''@aux: None/'action'/state' '''

        self.state_dim = state_dim
        self.env_action_dim = action_dim
        self.action_steps = action_steps
        self.dropout = dropout
        self.depthmap_mode = depthmap_mode

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
        self.bn = True
        self.img_dim = img_dim
        self.amp = amp
        self.reduced_dim = reduced_dim

        self.aux = aux
        if self.aux == 'action':
            self.aux_loss = nn.CrossEntropyLoss()
        elif self.aux == 'state':
            self.aux_loss = nn.MSELoss()
        self.aux_size = aux_size

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
                if self.aux is None:
                    n = QImage(action_dim=self.total_steps, img_stack=self.total_stack,
                        bn=self.bn, img_dim=self.img_dim,
                        drop=self.dropout).to(device)
                else:
                    if self.aux == 'action':
                        aux_size = self.total_steps
                    elif self.aux == 'state':
                        aux_size = self.aux_size

                    n = QImage2Outs(action_dim=self.total_steps, img_stack=self.total_stack,
                        bn=self.bn, img_dim=self.img_dim, drop=self.dropout,
                        aux_size=aux_size, reduced_dim=self.reduced_dim).to(device)
            elif self.network == 'densenet':
                n = QImageDenseNet(action_dim=self.total_steps,
                    img_stack=self.total_stack).to(device)
        elif self.mode == 'state':
            n = QState(state_dim=self.state_dim, action_dim=self.total_steps,
                    bn=self.bn, drop=self.dropout).to(device)
        elif self.mode == 'mixed':
            if self.network == 'simple':
                n = QMixed(state_dim=self.state_dim, action_dim=self.total_steps,
                    img_stack=self.total_stack, bn=self.bn,
                    img_dim=self.img_dim, drop=self.dropout).to(device)
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

        def handle_img(img):

            if img.ndim < 4:
                img = np.expand_dims(img, 0)
            if self.depthmap_mode:

                # unswizzle depth
                shape = img.shape
                depth = np.zeros((shape[0], 3, shape[2], shape[3]), dtype=np.int32)
                depth[:,0,:,:] = img[:,3,:,:]
                depth[:,1,:,:] = img[:,4,:,:]
                depth[:,2,:,:] = img[:,5,:,:]
                depth[:,0,:,:] |= depth[:,1,:,:] << 8
                depth[:,0,:,:] |= depth[:,2,:,:] << 16
                depth = depth[:,0,:,:]
                depth = np.expand_dims(depth, 1)
                depth = depth.astype(np.float32)
                depth /= e24

                # debug
                #plt.imsave('./depth_test.png', depth.squeeze()) # debug
                #debug_img = img[0, :3, :, :].transpose((1,2,0))
                #plt.imsave('./img_test.png', debug_img)

                depth = torch.from_numpy(depth).to(device)
                # Add onto state
                img = img[:,:4,:,:] # Only 4 dims
                img = torch.from_numpy(img).to(device).float()
                img /= 255.0

                #depth[0,0,0,0] = 0. #debug vs broadcasting
                img[:,3,:,:] = depth[:,0,:,:] # Overwrite with depth


            else:
                img = torch.from_numpy(img).to(device).float()
                img /= 255.0
            return img

        # Copy as uint8
        if self.mode == 'image':
            state = handle_img(state)
        elif self.mode == 'state':
            state = torch.from_numpy(state).to(device).float()
        elif self.mode == 'mixed':
            img = state[0]
            img = handle_img(img)
            state2 = torch.from_numpy(state[1]).to(device).float()
            state = (img, state2)
        else:
            raise ValueError('Unrecognized mode ' + mode)

        return state

    def _copy_sample_to_dev(self, x, y, u, r, d, best_action, extra_state, batch_size):
        x = self._process_state(x)
        y = self._process_state(y)
        # u is the actions: still as ndarrays
        u = u.reshape((batch_size, self.env_action_dim))
        # Convert continuous action to discrete
        u = self._cont_to_discrete(u).reshape((batch_size, -1))
        u = torch.LongTensor(u).to(device)
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        if best_action is not None:
            best_action = self._cont_to_discrete(best_action).reshape((batch_size, -1))
            best_action = torch.LongTensor(best_action).to(device)
        if extra_state is not None:
            extra_state = extra_state.reshape((batch_size, -1))
            extra_state = torch.FloatTensor(extra_state).to(device)
        return x, y, u, r, d, best_action, extra_state

    def _copy_sample_to_dev_small(self, x, best_action, extra_state, batch_size):
        x = self._process_state(x)
        best_action = self._cont_to_discrete(best_action).reshape((batch_size, -1))
        best_action = torch.LongTensor(best_action).to(device)
        extra_state = extra_state.reshape((batch_size, -1))
        extra_state = torch.FloatTensor(extra_state).to(device)
        return x, best_action, extra_state

    def select_action(self, state):

        state = self._process_state(state)
        q = self.q(state)

        if self.aux is not None:
            q = q[0]

        # Argmax along action dimension (not batch dim)
        max_action = torch.argmax(q, -1).cpu().data.numpy()

        # Translate action choice to continous domain
        action = self._discrete_to_cont(max_action)
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        # Sample replay buffer
        [x, y, u, r, d, best_action, extra_state, indices] = replay_buffer.sample(
            args.batch_size, beta=beta)

        length = len(u)

        state, state2, action, reward, done, best_action, extra_state = \
            self._copy_sample_to_dev(x, y, u, r, d, best_action, extra_state, length)

        Q_ts = self.q_target(state2)
        if self.aux is not None:
            Q_ts = Q_ts[0]
        Q_t = torch.max(Q_ts, dim=-1, keepdim=True)[0]

        # Compute the target Q value
        # done: We use reverse of done to not consider future rewards

        Q_t = reward + (done * args.discount * Q_t).detach()

        # Get current Q estimate
        Q_now = self.q(state)
        if self.aux is not None:
            Q_now, predicted = Q_now
        Q_now = torch.gather(Q_now, -1, action)

        # Compute Q loss
        q_loss = (Q_now - Q_t).pow(2)
        prios = q_loss + 1e-5
        prios = prios.data.cpu().numpy()
        q_loss = q_loss.mean()

        if self.aux is not None:
            compare_to = best_action if self.aux == 'action' else extra_state
            aux_loss = self.aux_loss(predicted, compare_to)
            aux_losses.append(aux_loss.item())

            self.q.freeze_some(False)

            self.q_optimizer.zero_grad()
            aux_loss.backward()
            self.q_optimizer.step()

            self.q.freeze_some(True) # Only backprop last layers

        # debug graph
        '''
        import torchviz
        dot = torchviz.make_dot(q_loss, params=dict(self.q.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

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

        a_ret = None
        if self.aux is not None:
            a_ret = aux_loss.item()

        return q_loss.item(), a_ret, Q_now.mean().item(), Q_now.max().item()

    def test(self, replay_buffer, args):
            [x, _, _, _, _, best_action, extra_state, _] = replay_buffer.sample(args.batch_size)
            length = len(x)

            state, action, extra_state = self._copy_sample_to_dev_small(x, best_action, extra_state, length)

            _, predict = self.q(state)
            if self.aux == 'action':
                predict = F.softmax(predict, dim=-1).argmax(dim=-1)
                y = action.cpu().data.numpy()
            elif self.aux == 'state':
                y = extra_state.cpu().data.numpy()

            predict = predict.cpu().data.numpy()

            return y, predict

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
        if self.aux is not None:
            q = q[0]

        max_action = torch.argmax(q, -1).cpu().data.numpy()

        action = self._discrete_to_cont(max_action)
        return action

    def train(self, replay_buffer, timesteps, beta, args):

        q_losses, aux_losses, Q_max, Q_mean = [], [], [], []

        for num, (update_q, update_qt, opt) in \
                enumerate(zip(self.qs, self.qts, self.opts)):

            # Get samples
            [x, y, u, r, d, best_action, extra_state, indices] = \
                replay_buffer.sample(args.batch_size, beta=beta, num=num)
            length = len(u)

            state, state2, action, reward, done, best_action, extra_state = \
                self._copy_sample_to_dev(x, y, u, r, d, best_action, extra_state, length)

            if self.aux is not None:
                update_q.freeze_some(False)

                _, predicted = update_q(state)
                compare_to = best_action if self.aux == 'action' else extra_state
                aux_loss = self.aux_loss(predicted, compare_to)
                aux_losses.append(aux_loss.item())

                opt.zero_grad()
                aux_loss.backward()
                opt.step()

                update_q.freeze_some(True) # Only backprop last layers


            Qt = [qt(state2) for qt in self.qts]
            if self.aux is not None:
                Qt = [q[0] for q in Qt]
            Qt = torch.min(*Qt)

            Qt_max, _ = torch.max(Qt, dim=-1, keepdim=True)

            y = reward + (done * args.discount * Qt_max).detach()

            # Get current Q estimate
            Q_now = update_q(state)
            if self.aux is not None:
                Q_now, _ = Q_now
            Q_now = torch.gather(Q_now, -1, action)

            # Compute loss
            q_loss = (Q_now - y).pow(2)
            prios = q_loss + 1e-5
            prios = prios.data.cpu().numpy()
            replay_buffer.update_priorities(indices, prios)
            q_loss = q_loss.mean()

            # Optimize the model
            opt.zero_grad()
            q_loss.backward()

            opt.step()

            # Update the frozen target models
            for p, pt in zip(update_q.parameters(), update_qt.parameters()):
                pt.data.copy_(args.tau * p.data + (1 - args.tau) * pt.data)

            q_losses.append(q_loss.item())
            Q_mean.append(Q_now.mean().item())
            Q_max.append(Q_now.max().item())

        aux_ret = None
        if self.aux is not None:
            aux_ret = np.mean(aux_losses)

        return np.mean(q_losses), aux_ret, np.mean(Q_mean), np.max(Q_max)

    def test(self, replay_buffer, args):
            [x, _, _, _, _, best_action, extra_state, _] = replay_buffer.sample(args.batch_size)
            length = len(x)

            state, action, extra_state = self._copy_sample_to_dev_small(x, best_action, extra_state, length)

            _, predict = self.qs[0](state)
            if self.aux == 'action':
                predict = F.softmax(predict, dim=-1).argmax(dim=-1)
                y = action.cpu().data.numpy()
            elif self.aux == 'state':
                y = extra_state.cpu().data.numpy()

            predict = predict.cpu().data.numpy()

            return y, predict

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

