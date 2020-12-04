import numpy as np
import torch as th
import torch.nn as nn
import os, sys, math
from os.path import join as pjoin

device = th.device("cuda" if th.cuda.is_available() else "cpu")

e24 = pow(2,24)

class OffPolicyAgent(object):
    def __init__(self, state_dim, action_dim, stack_size,
            mode, network:str, lr=1e-4, img_depth=3, img_dim=224,
            amp=False, dropout=False, aux:bool=False, aux_size=6, reduced_dim=10,
            depthmap_mode=False, freeze=False, opt='adam'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dropout = dropout
        self.depthmap_mode = depthmap_mode

        self.total_stack = stack_size * img_depth
        self.mode = mode
        self.network = network
        self.lr = lr
        self.bn = True
        self.img_dim = img_dim
        self.amp = amp
        self.reduced_dim = reduced_dim
        self.freeze = freeze
        self.opt_type = opt

        self.aux = aux
        if self.aux:
            self.aux_loss = nn.MSELoss()
        self.aux_size = aux_size

        self.to_save = []

    def _create_opt(self, model, lr):
        if self.opt_type == 'adam':
            print("opt: Adam. LR={}".format(lr))
            opt = th.optim.Adam(model.parameters(), lr=lr)
        elif self.opt_type == 'sgd':
            print("opt: SGD. LR={}".format(lr))
            opt = th.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError('Unknown optimizer type')

        return model, opt

    def _copy_action_to_dev(self, u, batch_size):
        return th.FloatTensor(u).to(device)

    def _copy_sample_to_dev(self, x, y, u, r, d, extra_state, batch_size):
        x = self._process_state(x)
        y = self._process_state(y)
        u = self._copy_action_to_dev(u, batch_size)
        r = th.FloatTensor(r).to(device)
        d = th.FloatTensor(1 - d).to(device)
        if extra_state is not None:
            extra_state = extra_state.reshape((batch_size, -1))
            extra_state = th.FloatTensor(extra_state).to(device)
        return x, y, u, r, d, extra_state

    def _sample_to_dev(self, replay_buffer, batch_size, beta, num):
        data = replay_buffer.sample(batch_size, beta=beta, num=num)
        [x, y, u, r, d, extra_state, indices] = data
        length = len(u)

        state, state2, action, reward, done, extra_state = \
            self._copy_sample_to_dev(x, y, u, r, d, extra_state, length)

        return state, state2, action, reward, done, extra_state, indices

    def _copy_sample_to_dev_small(self, x, extra_state, batch_size):
        x = self._process_state(x)
        extra_state = extra_state.reshape((batch_size, -1))
        extra_state = th.FloatTensor(extra_state).to(device)
        return x, extra_state

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

                depth = th.from_numpy(depth).to(device)
                # Add onto state
                img = img[:,:4,:,:] # Only 4 dims
                img = th.from_numpy(img).to(device).float()
                img /= 255.0

                #depth[0,0,0,0] = 0. #debug vs broadcasting
                img[:,3,:,:] = depth[:,0,:,:] # Overwrite with depth
            else:
                img = th.from_numpy(img).to(device).float()
                img /= 255.0
            return img

        # Copy as uint8
        if self.mode == 'image':
            state = handle_img(state)
        elif self.mode == 'state':
            state = th.from_numpy(state).to(device).float()
        elif self.mode == 'mixed':
            img = state[0]
            img = handle_img(img)
            state2 = th.from_numpy(state[1]).to(device).float()
            state = (img, state2)
        else:
            raise ValueError('Unrecognized mode ' + mode)
        return state

    def save(self, path):
        checkpoint = {}
        for s in self.to_save:
            obj = self.__dict__[s]
            if isinstance(obj, list) or isinstance(obj, tuple):
                checkpoint[s] = [o.state_dict() for o in obj]
            else:
                checkpoint[s] = obj.state_dict()

        th.save(checkpoint, os.path.join(path, 'checkpoint.pth'))

    def load(self, path):
        checkpoint = th.load(os.path.join(path, 'checkpoint.pth'))

        for s in self.to_save:
            if isinstance(self.__dict__[s], list) or isinstance(self.__dict__[s], tuple):
                for obj, robj in zip(self.__dict__[s], checkpoint[s]):
                    obj.load_state_dict(robj)
            else:
                self.__dict__[s].load_state_dict(checkpoint[s])
