from __future__ import unicode_literals
import os, sys, platform, math
from os.path import abspath
from os.path import join as pjoin
import signal
import glob, random, time, copy
import atexit
import subprocess as sp

import ast

# For shared memory
import posix_ipc, mmap
from ctypes import c_uint8, c_int, c_float, POINTER
import numpy as np

import scipy, scipy.misc
import skimage.transform as transform
import pygame # for writing on surface

import psutil # Install via pip
import getpass


# Import reward functions

from env.sim_env.reward import *

from .. import common_env

'''
We have a synchronous protocol with the simulator:
When sending anything, we note the value at shared_mem[0].
To detect if anything has been received, we compare to our value before sending

NOTE:
* For some reason, connecting to the viewer requires 2 instances of the
  environment. Create one, shut it down, and create it again. Not so for the
  runner.
'''

# Set paths to simulator
cur_sys = platform.system().lower()
path_sep = ':' if 'linux' in cur_sys or 'darwin' in cur_sys else ';'
cur_dir = os.path.dirname(abspath(__file__))

sim_path = abspath(pjoin(cur_dir, '..', '..', 'sim'))
ini_path = './sim_path.ini'
if os.path.exists(ini_path):
    with open(ini_path, 'r') as f:
        sim_path = abspath(f.read().rstrip())

isi_path = pjoin(sim_path, 'ISIH3DModuleBase', 'python')
os.environ['PYTHONPATH'] = (isi_path + path_sep +
  os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '')
sys.path.append(isi_path)

import ISISim.init_env
import ISISim.common_module_starter

from SimulationBase import SimulatorConnection
from SimulationBase.NetworkHelper import Action

# stats from analyzing, for normalization
g_state_mean_suture = np.array([0., 0., 0., 0., 0., 0., # arm1
  -0.02851895, -0.01229529, -0.03983802, 0.20836899, 0.49168926, .1693523, # arm2
  -1., -1., # jaws
  -0.10972147, -0.6796254,  -10.591319, 1.7183018, 0.1363969, -0.6918012, # needle
  -0.07567041, -0.6896427, -10.585802, # cur_target
  -0.13098323, -0.72658867, -10.531871 # next_target
  ])

g_state_std_suture = np.array([1., 1., 1., 1., 1., 1., # arm1 (just for div)
 1.3650687e-02, 1.6208388e-02, 1.4525504e-02, 2.9186731e-05, 9.6843083e-05, 7.5277233e-01, #arm2
 1., 1., #jaws (for div)
 2.8134113e-02, 4.2242080e-02, 1.9235633e-02, 7.3476839e-01, 6.6922742e-01, 2.4341759e-01, #needle
 1.0, 1.0, 1.0,
 1.0, 1.0, 1.0])

# Random number to avoid conflict with other running sims
random_num = random.randrange(10000)

shared_path = "/H3D"
SHARED_VAR_NUM = 3  # Variables before array
SHARED_VAR_SIZE = SHARED_VAR_NUM * 4  # size of variables

e24 = int(pow(2, 24))

class State(object):
    def __init__(self):
        pass

    def from_data(self, data):
        ''' Interpret incoming data '''

        index = 1 # Skip msg_id
        self.error = data[index]; index += 1

        # TODO: for some reason, we're only getting one arm data
        # The other arm is zeroed out

        #2 arms * (pos, orn) * x,y,z
        arm_dims = (2,2,3)
        arm_size = np.prod(arm_dims)
        self.arm = np.array(data[index:index + arm_size],
            dtype=np.float32).reshape(arm_dims)
        index += arm_size

        # 2 jaws
        self.jaw = data[index:index + arm_dims[0]]
        index += arm_dims[0]

        # World coordinates of jaw tips
        tip_dims = (2,2,3)
        tip_size = np.prod(tip_dims)
        self.tip_state = np.array(data[index:index + tip_size],
            dtype=np.float32).reshape(tip_dims)
        index += tip_size

        # Needle state
        needle_dims = (2,3) # pos, orn
        needle_size = np.prod(needle_dims)
        self.needle = np.array(data[index:index + needle_size],
            dtype=np.float32).reshape((2,3))
        index += needle_size

        # Get 10 (or more) points of needle. First one is tip
        num_points = data[index]; index += 1
        self.needle_points_pos = np.array(data[index:index + num_points * 3],
            dtype=np.float32).reshape((num_points, 3))
        index += num_points * 3
        self.needle_tip_pos = self.needle_points_pos[0]

        ## From NeedleDriving.py ##
        self.curvature_radius = data[index]; index += 1
        self.needle_grasped = data[index]; index += 1
        self.needle_insert_status = data[index]; index += 1
        self.num_targets = data[index];    index += 1
        self.cur_target = data[index]; index += 1
        # 0 = new throw, 1 = inserted, not exited
        # 2 = inserted, exited, 3 = post 2, needle removed from entrance target
        self.target_insert_status = data[index]; index += 1

        # Next target location
        self.cur_target_pos = np.array(data[index: index + 3], dtype=np.float32);  index += 3
        self.next_target_pos = np.array(data[index: index + 3], dtype=np.float32); index += 3

        self.tools_out_of_view = data[index]; index += 1
        self.instr_collisions = data[index]; index += 1
        self.instr_endo_collisions = data[index]; index += 1
        self.endo_env_collisions = data[index]; index += 1
        self.total_motion = data[index]; index += 1
        self.excessive_force = data[index]; index += 1
        self.excessive_needle_pierces = data[index]; index += 1
        self.excessive_insert_needle_pierces = data[index]; index += 1
        self.excessive_exit_needle_pierces = data[index]; index += 1
        self.excessive_needle_tissue_force = data[index]; index += 1
        self.needle_tip_grabbed = data[index]; index += 1
        self.incorrect_needle_throws = data[index]; index += 1

        # Handle older data
        if len(data) > index:
            self.tissue_corners = []
            self.tissue_corners.append(np.array(data[index:index+3], dtype=np.float32)); index += 3
            self.tissue_corners.append(np.array(data[index:index+3], dtype=np.float32)); index += 3
            self.tissue_corners.append(np.array(data[index:index+3], dtype=np.float32)); index += 3

        if len(data) > index:
            self.outside_insert_radius = data[index]; index += 1
            self.outside_exit_radius = data[index]; index += 1

class Environment(common_env.CommonEnv):

    def __init__(self, mode='state', start_port=50001,
        start_env_port=50002,
        stack_size=1, img_dim=224, program='runner', max_steps=100,
        random_target=False, task='reach',
        hi_res_mode=False, stereo_mode=False, depthmap_mode=False, full_init=True,
        reward='simple', cnn_test_mode=False, *args, **kwargs):
        '''
        @server_num: which number environment this is
        '''

        super(Environment, self).__init__(**kwargs)

        self.program = program
        self.sim_pid = None
        self.sim_connection = None
        self.shared_path = shared_path + str(random_num) + str(self.server_num)
        self.port = start_port + random_num + self.server_num * 2
        self.env_port = start_env_port + random_num + self.server_num * 2
        self.msg_id = 0
        self.rcv_msg_id = -1
        print("XXX port=", self.port, " env_port=", self.env_port) # debug

        # For shared memory support
        self.shared_mem = None
        self.mmap_shared = None
        self.shared_vars = [0]
        self.shared_rgb = None
        self.ready_flag_old = 0
        self.img_count = 1
        self.last_event = None
        self.task = task
        self.reward_type = reward
        self.random_target = random_target

        # We always read/save hi-res and stereo, but we feed the rl algo
        # low-res and mono/stereo
        self.hi_res_mode = hi_res_mode
        self.stereo_mode = stereo_mode
        self.depthmap_mode = depthmap_mode

        # CNN Testing mode
        self.cnn_test_mode = cnn_test_mode

        # How often to reset the environment
        # Due to memory leaks or just errors
        self.reboot_eps = 300
        if self.get_save_mode() == 'play':
            self.reboot_eps = 1000000
        # Reboot after these many errors
        self.max_error_ctr = 7
        self.error_ctr = 0

        self.episode = 0
        self.total_time = 0
        self.mode = mode
        self.img_dim = img_dim
        self.stack_size = stack_size
        #self.resolution = (1024 * 2, 1024)
        self.resolution = (1000, 800)

        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 10)

        # We can only change pos or orn at a time
        self.max_steps = max_steps

        self.state = None
        self.last_state = None

        # Reward function
        if task == 'reach':
            self.reward = Reward_reach_v0(self)
            # pos/rot(3)
            self.action_steps = np.array([3, 3, 3])
        elif task == 'suture':
            if self.reward_type == 'simple':
                self.reward = Reward_suture_simple(self)
            elif self.reward_type == 'v1':
                self.reward = Reward_suture_v0(self)
            else:
                raise ValueError('Unrecognized reward: ' + reward)

            # selector, pos/rot(3)
            self.action_steps = np.array([3, 3, 3, 3])
        else:
            raise ValueError("Unknown task " + task)

        self.action_dim = len(self.action_steps)

        self.extra_state_dim = 6
        self.render_ep_path = None


        if full_init:
            # Create sim_connection
            self.sim_connection = SimulatorConnection.SimulatorConnection(
                'localhost', self.port, via_daemon=False)

            if not self._is_running():
                self._reboot_until_up()

    def _reboot_until_up(self):
        ''' Reboot the sim and reconnect to it '''
        good = False
        while not good:
            self._kill_sim()
            time.sleep(6)
            good = self._connect_to_sim()
            self.clean_up_env()  # Remove zombies

    def _kill_sim(self):
        if self.sim_pid is not None:
            os.kill(self.sim_pid, signal.SIGTERM)
            self.sim_pid = None

    # Kill sim on exit
    def __del__(self):
        self._kill_sim()
        self.sim_connection = None

    def _callAndCheckAction(self, name, data=[]):
        # Make space-separated lists
        def get_str(ds):
            if isinstance(ds, list):
                r = ' '.join(get_str(d) for d in ds)
            else:
                r = str(ds)
            return r

        self.ready_flag_old = self.shared_vars[0]
        self.msg_id += 1
        #print("sending msg_id ", self.msg_id) #debug

        # List of items -> list of strings
        if isinstance(data, list):
            data = [str(self.msg_id)] + [get_str(d) for d in data]
        else:
            data = [str(self.msg_id), str(data)]

        #print("callAndCheckAction: {}, sending {}".format(name, data)) # debug
        action = Action(name, '', data)
        res = self.sim_connection.call(action)
        if not res.isSuccess():
            s = 'There was an error executing the action ' + name + ':'
            s += '\nDescription: ' + res.getDescription() + '\n'
            print(s)
        return res

    def _formatActionResult(self, name, action_result):
          if not action_result.isSuccess():
              s = "There was an error executing the action:"
              s += "\nDescription: " + action_result.getDescription() + "\n"
              print(s)

    def _read_shared_data(self, save_img=False):
        width = self.shared_vars[1]
        height = self.shared_vars[2]
        self.resolution = (width, height)
        #print("XXX width = ", width, "height = ", height) # debug

        # DEBUG
        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        # array of uint8s
        rgb = np.ctypeslib.as_array(self.shared_rgb)
        rgb = np.reshape(rgb, (height, width, 3))
        rgb = np.flipud(rgb)
        # Depth is expressed as a float
        depth = np.ctypeslib.as_array(self.shared_depth)
        depth = np.reshape(depth, (height, width, 1))
        depth = np.flipud(depth)

        if save_img:
            scipy.misc.imsave('./img{}_{}.png'.format(
              self.server_num, self.img_count), rgb)
            self.img_count += 1

        return rgb, depth

    def _init_shared_mem(self):
        # open shared mem
        if self.shared_mem is None:
            self.shared_mem = posix_ipc.SharedMemory(self.shared_path)
            self.mmap_shared = mmap.mmap(self.shared_mem.fd, 0)

            #from rl.utils import ForkablePdb
            #ForkablePdb().set_trace()

            self.shared_vars = (c_int * SHARED_VAR_NUM).from_buffer(self.mmap_shared)
            width, height = self.resolution
            s_type = c_uint8 * (width * height * 3)
            self.shared_rgb = s_type.from_buffer(self.mmap_shared, SHARED_VAR_SIZE) # offset
            s_type2 = c_float * (width * height)
            self.shared_depth = s_type2.from_buffer(self.mmap_shared, SHARED_VAR_SIZE + width * height * 3)

    def _callback(self, event):
        # Save event
        event = event.getContentText()
        event = ast.literal_eval(event)
        self.rcv_msg_id = event[0]
        self.last_event = event

        # If first time for callback, open shared mem
        if self.shared_mem is None:
            self._init_shared_mem()

    def _is_running(self):
        return self.sim_pid is not None

    def _connect_to_sim(self):
        self.program = 'runner'

        # Launch sim
        path = pjoin(sim_path, 'ISIH3DModuleBase', 'modules', 'NeedleDriving')
        self.sim_pid = ISISim.common_module_starter.startModule(
            'NeedleDriving', 1, 'debug', 'runner', path,
            resolution=self.resolution, random_num=random_num,
            server_num=self.server_num)

        time.sleep(6)

        # Create sim_connection
        #self.sim_connection = SimulatorConnection.SimulatorConnection(
        #    'localhost', self.port, via_daemon=False)

        # Loop until sim is up
        success = False
        while not success:

            # Get the simulation version string
            result = self.sim_connection.call(Action('getAPIName'))
            if result.isSuccess():
                success = True
                apiName = result.getContentText()
            else:
                print('Could not connect to network API.')
                self.error_ctr += 1
                if self.error_ctr > self.max_error_ctr:
                    return False
                time.sleep(6)

        print('Connected to ' + apiName)
        self.error_ctr = 0

        # Create a callback and tell the simulation about it (event)
        callback_id = self.sim_connection.registerSimpleEvent(self._callback)
        result = self.sim_connection.startCallbackListener(
            'localhost', self.env_port)
        if not result.isSuccess():
            print('ERROR: SimulatorConnection::startCallbackListener' \
                'could NOT be performed successfully.')
            print(result.getDescription())
            print(result.getContentText())
            sys.exit()

        # Set the callback id in the simulator
        self._callAndCheckAction('setEventId', callback_id)

        self._callAndCheckAction('initSim')

        self._wait_for_sim()

        # Now our connection to the sim is complete
        return True

    def render(self, save_path='./out', sim_save=False, text=''):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write reward on image
        # pygame is (x,y) while our array is (y,x)

        img = self.image.transpose((1,0,2))
        surface = pygame.surfarray.make_surface(img)

        if not sim_save:
            n = self.state.needle_tip_pos
            t = self.state.cur_target_pos
            reward_s = 'S{} TR:{:.3f}  R:{:.3f}  EP:{} d:{:.4f}'.format(
                  self.server_num, self.total_reward, self.last_reward,
                  self.episode, self.reward.last_dist)

            if self.task == 'reach':
                reward_s += ' n:({:.2f},{:.2f},{:.2f}) t:({:.2f},{:.2f},{:.2f})'.format(
                    n[0], n[1], n[2], t[0], t[1], t[2])

            elif self.task == 'suture':
                if self.reward_type == 'v1':
                    reward_s += ' ai:{:.2f}, di:{:.2f}'.format(
                        float(self.reward.last_a_ideal), float(self.reward.last_dist_ideal))
                reward_s += ' ns:{}, ts:{}, {}'.format(
                    self.state.needle_insert_status, self.state.target_insert_status,
                    text)
                n = self.state.needle_tip_pos
                t = self.state.cur_target_pos
                reward_s += ' n:({:.2f},{:.2f},{:.2f}) t:({:.2f},{:.2f},{:.2f})'.format(
                    n[0], n[1], n[2], t[0], t[1], t[2])

                reward_s += ' ' + self.reward_txt
            try:
                txt_surface = self.font.render(reward_s, False, (255,0,0))
            except:
                print("reward_s = ", reward_s)
                raise

            surface.blit(txt_surface, (10,10))

        save_name = self._gen_save_name(sim_save=sim_save) + '.png'
        save_file = pjoin(save_path, save_name)
        pygame.image.save(surface, save_file)

    def convert_to_video(self, save_path='./out', sim_save=False):
        ''' Convert to video and delete image files '''
        save_name = self._gen_save_name(sim_save=sim_save, use_t=False)
        pattern = pjoin(save_path, save_name + '*.png')
        out_file = pjoin(save_path, save_name + '.mp4')
        filelist = glob.glob(pattern)
        if len(filelist) > 0:
            cmd = 'cat {} | ffmpeg -loglevel 8 -f image2pipe -r 5 -vcodec png -i - -vcodec libx264 -preset veryslow {}'.format(pattern, out_file)
            os.system(cmd)
            for f in filelist:
                os.remove(f)
        else:
            print("convert_to_video: No files found for ", pattern)

    def _wait_for_sim(self):
        # Wait for reply: both the packet and the shared mem
        # debug
        #print("rcv_msg_id is {}, ready_flag is {}".format(
        #    self.rcv_msg_id, self.shared_vars[0]))

        while (self.rcv_msg_id < self.msg_id or
               self.ready_flag_old == self.shared_vars[0]):
            # debug
            #print("rcv_msg_id is {}, ready_flag is {}".format(
            #    self.rcv_msg_id, self.shared_vars[0]))
            time.sleep(0.04)

    def _update_sim_state(self, action=None):

        self._wait_for_sim()

        image, depth = self._read_shared_data()

        # Resize/crop image (always stereo)
        x = [0., 1.]
        y = [0.40, 0.82] # Get relevant part of image
        cropx = [int(self.resolution[0] * x[0]), int(self.resolution[0] * x[1]) + 1]
        cropy = [int(self.resolution[1] * y[0]), int(self.resolution[1] * y[1]) + 1]
        image = image[cropy[0]:cropy[1], cropx[0]:cropx[1], :]

        w, h = self._get_width_height(hires=True, stereo=True, depth=False)
        # Transform Converts to float64 and 0 to 1.0
        self.image = transform.resize(image, (h,w), anti_aliasing=False)
        self.image *= 255.0
        self.image = self.image.astype(np.uint8)

        # Crop depth to just the left image
        x = [0., 0.5]
        cropx = [int(self.resolution[0] * x[0]), int(self.resolution[0] * x[1]) + 1]
        depth = depth[cropy[0]:cropy[1], cropx[0]:cropx[1], :]

        # no interpolation on depth
        w, h = self._get_width_height(hires=True, stereo=False, depth=False)
        # Transform converts to float64
        depth = transform.resize(depth, (h,w), anti_aliasing=False)
        depth *= e24 # max value
        # Separate out components
        depth = depth.astype(int).squeeze(-1)
        depth2 = np.zeros((h, w, 3), dtype=np.uint8)
        depth2[:,:,0] = depth & 0xFF
        depth2[:,:,1] = (depth >> 8) & 0xFF
        depth2[:,:,2] = (depth >> 16) & 0xFF

        self.image = np.concatenate([self.image, depth2], axis=1)

        event = self.last_event
        self.last_event = None

        # Backup old state, read all the state in the event
        self.last_state = self.state
        self.state = State()

        self.state.from_data(event)
        self.state.action = action
        #print("_update_sim_state: action = ", self.state.action) # debug

    def _save_img(self, img):
        scipy.misc.imsave('./out/img{}_{}.png'.format(
          self.server_num, self.img_count), img)
        self.img_count += 1

    def _get_best_action(self):
        # Simple movement for best action
        # Get motion needed
        diff = -(self.state.needle_tip_pos - self.state.cur_target_pos)
        epsilon = 0.03
        for i in range(3):
          if diff[i] < -epsilon:
            diff[i] = -1.
          elif diff[i] > epsilon:
            diff[i] = 1.

        return np.reshape(diff, (3,))

    def _get_extra_state(self):
        ''' Get extra state for aux loss
            Make sure to normalize or it messes up losses
        '''
        nt = self.state.needle_tip_pos
        tar = self.state.cur_target_pos

        # Normalizing: very important for performance!
        if self.task == 'reach':
            nt -= np.array([-0.02, -0.47, -10.82])
            nt /= np.array([0.12, 0.06, 0.08])
            tar -= np.array([-0.18, -0.59, -10.71])
            tar /= np.array([0.17, 0.11, 0.13])
        elif self.task == 'suture':
            nt -= np.array([-0.12, -0.73, -10.53])
            nt /= np.array([0.06, 0.04, 0.06])
            if tar.dtype == 'int64':
                print ('Error in tar: got int64: ', tar)
                tar = np.array([0., 0., 0.], dtype=np.float32)
            else:
                tar -= np.array([-0.08, -0.69, -10.53])
                tar /= np.array([0.01, 0.01, 0.05])

        return np.concatenate([
          nt.reshape((1, -1)),
          tar.reshape((1, -1))
        ], axis=-1)


    def _get_env_state(self):
        image = self.image
        # Resize to non-hires
        w, h = self._get_width_height(hires=False, stereo=True, depth=True)
        image = transform.resize(self.image, (h,w), anti_aliasing=False)
        image *= 255.0
        image = image.astype(np.uint8)

        if self.depthmap_mode:
            # Process depthmap_mode into 6 layers
            # We'll later stitch it into 4
            # h = w / 3
            image_l = image[:, :h, :]
            depth = image[:, (h*2):, :]
            image = np.concatenate([image_l, depth], axis=2)
        elif self.stereo_mode:
            # Process stereo into 6 layers
            # h = w / 3
            image_l = image[:, :h, :]
            image_r = image[:, h:(h*2), :]
            image = np.concatenate([image_l, image_r], axis=2)
        else:
            image = image[:, :h, :]

        #debug
        #self._save_img(image)

        image = image.transpose((2,0,1)) # prepare for pytorch

        if self.mode == 'image':
            cur_state = np.expand_dims(image, 0)
        elif self.mode == 'state':
            cur_state = np.concatenate([
                self.state.arm.reshape((1,-1)),
                np.array(self.state.jaw, dtype=np.float32).reshape((1, -1)),
                self.state.needle.reshape((1, -1)),
                np.array(self.state.cur_target_pos, dtype=np.float32).reshape((1, -1)),
                np.array(self.state.next_target_pos, dtype=np.float32).reshape((1, -1)),
            ], axis=-1)

            # Normalize for suture task
            if self.task == 'suture':
                cur_state -= g_state_mean_suture
                cur_state /= g_state_std_suture

        elif self.mode == 'mixed':
            cur_state = (
                np.expand_dims(image, 0),
                np.concatenate([
                    self.state.arm.reshape((1,-1)),
                    np.array(self.state.jaw, dtype=np.float32).reshape((1,-1)),
                ], axis=-1)
            )
        else:
            raise ValueError("Unknown mode " + self.mode)

        return cur_state

    def _move_arm_one(self, arm, move_choice, data=[]):
        if move_choice == 'p':
          data = [arm, data, [0.,0.,0.], 0]
        elif move_choice == 'o':
          data = [arm, [0.,0.,0.], data, 0]
        elif move_choice == 'j':
          data = [arm, [0., 0., 0.], [0.,0.,0.], -1]
        elif move_choice == 'J':
          data = [arm, [0., 0., 0.], [0.,0.,0.], 1]

        self._callAndCheckAction('moveArm', data)
        self._wait_for_sim()

    def _step_real_sim(self, a_orig):
        ''' Take a real step in the sim '''

        if not self._is_running():
            self._reboot_until_up()

        # Real sim
        a = np.copy(a_orig)

        if self.task == 'reach':
            # Reach expects a 3-member action
            # 0,0,0 isn't accepted
            if np.all(a == 0):
                a[1] = 1.

            # vector, rotation, angle
            # Scale actions to the ranges we want
            arm = 2 # arm is fixed: right
            a *= 0.001 # pos
            pos = a.tolist()
            orn = [0., 0., 0.]

        elif self.task == 'suture':
            # Suture expects a 4-member action (1st is dummy)
            arm = 2

            # Help algorithm reach target
            if self.state.target_insert_status == 0:
                a[3] = 0.

            # 0,0,0 isn't accepted
            if np.all(a == 0):
                a[0] = 1.

            pos = (a[0:3] * 0.001).tolist()
            orn = [0., 0., a[3]*0.1]
        else:
            raise ValueError("Invalid task " + str(self.task))

        action_l = [arm, pos, orn, 0] # last is jaw

        self._callAndCheckAction('moveArm', action_l)

        self._update_sim_state(action=a_orig)

    # Action space:
    # one arm only, Position (3), orientation (3)
    def step(self, action_orig):

        if self.done:
            print("[{}] In step: Need reset".format(self.server_num))
            return

        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        self.t += 1
        self.total_time += 1

        play_done = False
        if self.get_save_mode() == 'play':
            # Dummy playback. Overwrite action
            action_orig, play_done = self._step_play()
        else:
            self._step_real_sim(action_orig)

        cur_state = self._get_env_state()
        reward, done, self.reward_txt, success = self.reward.get_reward_data()
        done = done or play_done

        self.done = done
        self.last_reward = reward
        self.total_reward += reward

        # Save the img/state action of every step
        # Upon reset, we'll dump the states file
        self._step_record()

        # Handle epsiode render if needed
        if self.render_ep_path is not None:
            self.render(self.render_ep_path, sim_save=False)

        extra = {"action": action_orig, "save_mode":self.get_save_mode(), "success":success}

        extra["extra_state"] = self._get_extra_state()

        return (cur_state, reward, done, extra)

    def _reset_real(self):

        if self.episode % self.reboot_eps == 0:
            print("Episode is ", self.episode)
            self._reboot_until_up()

        if self.error_ctr >= self.max_error_ctr:
            print("Error ctr is ", self.error_ctr)
            self._reboot_until_up()

        self.error_ctr = 0

        # Loop until we get a proper reset
        good = False
        while not good:
            if self.error_ctr > 5:
                self._reboot_until_up()
                self.error_ctr = 0

            # Load the state
            self.sim_load_state_cmd('reset_{}'.format(self.task), use_file=True)

            self._update_sim_state(action=None)

            good = True
            if not self.state.needle_grasped:
                good = False
            if self.state.needle_insert_status != 0:
                good = False
            self.error_ctr += 1

    def _get_width_height(self, hires=False, stereo=False, depth=False):
        img_dim = self.img_dim * 2 if hires else self.img_dim
        w = img_dim
        if stereo and depth:
          w *= 3
        elif stereo or depth:
          w *= 2
        h = img_dim
        return w, h

    def reset(self, render_ep_path=None):

        # Check if last episode was a record. If so, convert
        if self.render_ep_path is not None:
            self.convert_to_video(self.render_ep_path, sim_save=False)

        self.render_ep_path=render_ep_path
        self.t = 0
        self.episode += 1
        self.done = False
        self.insert_status = 0
        self.total_reward = 0.
        self.last_reward = 0.
        self.last_target = None
        self.state = None
        self.last_state = None
        self.image = None

        w, h = self._get_width_height(hires=True, stereo=True, depth=True)
        if not self._reset_try_play(w, h):
            self._reset_real()

        self._reset_record()

        cur_state = self._get_env_state()

        # Things required for the reward function
        self.reward.reset()

        # Handle episode render if needed
        if self.render_ep_path is not None:
            self.render(self.render_ep_path, sim_save=False)

        return (cur_state, 0, False, {"action": None, "save_mode": self.get_save_mode(), "success":False})

    @staticmethod
    def clean_up_env():
        # Check if a sim is running
        processes = psutil.process_iter()
        for p in processes:
            if 'h3drunner' in p.name().lower() and \
                p.username() == getpass.getuser() and \
                p.ppid() == 1: # zombie
                  print("Cleaning up pid", p.pid)
                  os.kill(p.pid, signal.SIGTERM)


    def combine_states(self, states):
        if self.mode == 'mixed':
            # list of tuples of ndarrays -> tuple of lists of ndarrays
            states = zip(*states)
            # -> tuple of ndarrays
            return [np.concatenate(s) for s in states]
        else:
            return np.concatenate(states)

    def sim_save_state_cmd(self, state='0'):
        # Save the current state as slot 0 for reset
        self._callAndCheckAction('saveState', state)
        self._wait_for_sim()

    def sim_load_state_cmd(self, state='0', use_file=True):
        self._callAndCheckAction('loadState', [state, use_file, self.random_target])
        self._wait_for_sim()

if __name__ == '__main__':
    import argparse
    from prompt_toolkit import prompt
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", dest="port", default=60001, type=int,
                      help="The port to connect to" )
    parser.add_argument("--envport", dest="envport", default=60005, type=int,
                      help="The port to use to listen for upstream events" )
    parser.add_argument("--prog", default='runner',
                      help="The program to run/connect to" )
    args = parser.parse_args()

    env = Environment(program=args.prog)

    while True:
        text = prompt('> ')
        if len(text) == 0:
            continue
        if text[0] == 'r':
            env.reset()
            print("here")
        elif text[0] == 's':
            name = text[2:]
            env.save_state(name)
        elif text[0] == 'l':
            name = text[2:]
            env.load_state(name)
        elif text[0] in '123':
            arm = int(text[0])
            if text[2] in 'jJ':
                env._move_arm_one(arm, text[2])
            elif text[2] in 'op':
                data = text[4:].split()
                data = [float(x) for x in data]
                env._move_arm_one(arm, text[2], data)
        else:
           print("Commands: r(eset), s(ave), l(oad), 1/2/3")

