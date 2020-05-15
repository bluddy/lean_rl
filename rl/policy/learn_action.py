import numpy as np
import torch
import torch.nn as nn
import os, sys, math
import torch.nn.functional as F
from os.path import join as pjoin
from models import QState, QImage, QMixed

''' A 'policy' that learns a correspondence between state and action '''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# learn-action

# We have greyscale, and then one RGB

class LearnAction(object):
    def __init__(self, use_extra_state, state_dim, action_dim, action_steps, stack_size,
            mode, network, lr=1e-4, img_depth=3, bn=True, img_dim=224,
            amp=False, deep=False, dropout=False, extra_state_dim=9):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_extra_state = use_extra_state
        self.extra_state_dim = extra_state_dim
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
        self.amp = amp

        if use_extra_state:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self._create_models()

        if self.amp:
            import amp

        print "LR=", lr

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def _create_model(self):
        if self.mode == 'image':
            if self.network == 'simple':
                action_dim = self.total_steps
                if self.use_extra_state:
                    action_dim = self.extra_state_dim
                n = QImage(action_dim=action_dim, img_stack=self.total_stack,
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
        self.model = self._create_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.amp:
            self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level='O1')

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

    def _copy_sample_to_dev(self, x, a, es, batch_size):
        x = self._process_state(x)
        a = a.reshape((batch_size, self.action_dim))
        # Convert continuous action to discrete
        a = self._cont_to_discrete(a).reshape((batch_size, -1))
        a = torch.LongTensor(a).to(device)
        es = es.reshape((batch_size, -1))
        es = torch.FloatTensor(es).to(device)
        return x, a, es

    def train(self, replay_buffer, args):

        # Sample replay buffer
        #import pdb
        #pdb.set_trace()
        [x, ba, es] = replay_buffer.sample(args.batch_size)

        length = len(a)

        state, best_action, extra_state = self._copy_sample_to_dev(x, ba, es, length)

        if self.use_extra_state:
            predicted_state = self.model(state)
            loss = self.loss(predicted_state, extra_state)
        else:
            predicted_action = self.model(state)
            loss = self.loss(predicted_action, action.squeeze(-1))

        # debug graph
        '''
        import torchviz
        dot = torchviz.make_dot(q_loss, params=dict(self.model.named_parameters()))
        dot.format = 'png'
        dot.render('graph')
        sys.exit(1)
        '''

        # Optimize the model
        self.optimizer.zero_grad()

        if self.amp:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.optimizer.step()

        return loss.item()

    def test(self, replay_buffer, args):
            [x, ba, es] = replay_buffer.sample(args.batch_size)
            length = len(ba)

            state, best_action, extra_state = self._copy_sample_to_dev(x, ba, es, length)

            x = self.model(state)

            if self.use_extra_state:
                x = x.cpu().data.numpy().flatten()
                y = extra_state.cpu().data.numpy().flatten()
            else:
                x = F.softmax(x, dim=-1).argmax(dim=-1)
                x = x.cpu().data.numpy().flatten()
                y = best_action.cpu().data.numpy().flatten()

            return y, x

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))

    def load(self, path):
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))

