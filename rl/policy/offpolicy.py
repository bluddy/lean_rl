import numpy as np
import torch
import torch.nn as nn
import os, sys, math
from os.path import join as pjoin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import apex.amp as amp
except ImportError:
    pass

# DQN

e24 = pow(2,24)

class OffPolicyAgent(object):
    def __init__(self, state_dim, action_dim, stack_size,
            mode, network:str, lr=1e-4, img_depth=3, img_dim=224,
            amp=False, dropout=False, aux:bool=False, aux_size=6, reduced_dim=10,
            depthmap_mode=False, freeze=False, opt='adam'):

        self.state_dim = state_dim
        self.env_action_dim = action_dim
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

    def _copy_sample_to_dev(self, x, y, u, r, d, extra_state, batch_size):
        x = self._process_state(x)
        y = self._process_state(y)
        u = self._copy_action_to_dev(u)
        r = torch.FloatTensor(r).to(device)
        d = torch.FloatTensor(1 - d).to(device)
        if extra_state is not None:
            extra_state = extra_state.reshape((batch_size, -1))
            extra_state = torch.FloatTensor(extra_state).to(device)
        return x, y, u, r, d, extra_state

    def _copy_sample_to_dev_small(self, x, extra_state, batch_size):
        x = self._process_state(x)
        extra_state = extra_state.reshape((batch_size, -1))
        extra_state = torch.FloatTensor(extra_state).to(device)
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
