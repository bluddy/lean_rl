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
from ctypes import c_uint8, c_int, POINTER
import numpy as np

import scipy, scipy.misc
import pygame # for writing on surface

import psutil # Install via pip
import getpass


# Import reward functions

from reward import *

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
isi_path = pjoin(sim_path, 'ISIH3DModuleBase', 'python')
os.environ['PYTHONPATH'] = (isi_path + path_sep +
  os.environ['PYTHONPATH'] if 'PYTHONPATH' in os.environ else '')
sys.path.append(isi_path)

import ISISim.init_env
import ISISim.common_module_starter

from SimulationBase import SimulatorConnection
from SimulationBase.NetworkHelper import Action


# Random number to avoid conflict with other running sims
random_num = random.randrange(10000)

shared_path = "/H3D"
SHARED_VAR_NUM = 3  # Variables before array
SHARED_VAR_SIZE = SHARED_VAR_NUM * 4  # size of variables

EXPERT_STRATEGY = False

class State(object):
    def __init__(self):
        pass

    def from_data(self, data):
        ''' Interpret incoming data '''

        index = 1 # Skip msg_id
        self.error = data[index]; index += 1

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
        self.cur_target_pos = np.array(data[index: index + 3]);  index += 3
        self.next_target_pos = np.array(data[index: index + 3]); index += 3

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

class Environment(common_env.CommonEnv):

    def __init__(self, mode='state', start_port=50001,
        start_env_port=50002,
        stack_size=1, img_dim=224, program='runner', max_steps=100,
        random_target=False, task='reach',
        hi_res_mode=False, stereo_mode=False, full_init=True,
        *args, **kwargs):
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
        print "XXX port=", self.port, " env_port=", self.env_port # debug

        # For shared memory support
        self.shared_mem = None
        self.mmap_shared = None
        self.shared_vars = [0]
        self.shared_array = None
        self.ready_flag_old = 0
        self.img_count = 1
        self.last_event = None
        self.task = task
        self.random_target = random_target

        self.hi_res_mode = hi_res_mode
        self.stereo_mode = stereo_mode

        # How often to reset the environment
        # Due to memory leaks or just errors
        self.reboot_eps = 300
        # After 5 errors, reboot
        self.max_error_ctr = 5
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
            self.reward = Reward_suture_v0(self)
            # selector, pos/rot(3)
            self.action_steps = np.array([2, 3, 3, 3, 3])
        else:
            raise ValueError("Unknown task " + task)

        self.action_dim = len(self.action_steps)

        if full_init:
            self._connect_to_sim()

    def _reboot(self):
        ''' Reboot the sim and reconnect to it '''
        good = False
        while not good:
            print '[{}] Rebooting'.format(self.server_num)
            self._kill_sim()
            time.sleep(6)
            good = self._connect_to_sim()

    def _kill_sim(self):
        self.sim_connection = None
        if self.sim_pid is not None:
            os.kill(self.sim_pid, signal.SIGTERM)

    # Kill sim on exit
    def __del__(self):
        self._kill_sim()

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
        #print "sending msg_id ", self.msg_id #debug

        # List of items -> list of strings
        if isinstance(data, list):
            data = [str(self.msg_id)] + [get_str(d) for d in data]
        else:
            data = [str(self.msg_id), str(data)]

        #print "callAndCheckAction: {}, sending {}".format(name, data) # debug
        action = Action(name, '', data)
        res = self.sim_connection.call(action)
        if not res.isSuccess():
            s = 'There was an error executing the action ' + name + ':'
            s += '\nDescription: ' + res.getDescription() + '\n'
            print s
        return res

    def _formatActionResult(self, name, action_result):
          if not action_result.isSuccess():
              s = "There was an error executing the action:"
              s += "\nDescription: " + action_result.getDescription() + "\n"
              print s

    def _read_shared_data(self, save_img=False):
        width = self.shared_vars[1]
        height = self.shared_vars[2]
        self.resolution = (width, height)
        #print "XXX width = ", width, "height = ", height # debug
        arr = np.ctypeslib.as_array(self.shared_array)
        arr = np.reshape(arr, (height, width, 3))
        arr = np.flipud(arr)

        if save_img:
            scipy.misc.imsave('./img{}_{}.png'.format(
              self.server_num, self.img_count), arr)
            self.img_count += 1

        return arr

    def _init_shared_mem(self):
        # open shared mem
        if self.shared_mem is None:
            self.shared_mem = posix_ipc.SharedMemory(self.shared_path)
            self.mmap_shared = mmap.mmap(self.shared_mem.fd, 0)
            self.shared_vars = (c_int * SHARED_VAR_NUM).from_buffer(
                self.mmap_shared)
            s_type = c_uint8 * (self.mmap_shared.size() - SHARED_VAR_SIZE)
            self.shared_array = s_type.from_buffer(self.mmap_shared,
                SHARED_VAR_SIZE)

    def _callback(self, event):
        # Save event
        event = event.getContentText()
        event = ast.literal_eval(event)
        self.rcv_msg_id = event[0]
        self.last_event = event

        # If first time for callback, open shared mem
        if self.shared_mem is None:
            self._init_shared_mem()

    def _connect_to_sim(self):
        # First check for a running viewer
        #sim_proc_list = [p[0] for p in processes if 'h3dviewer' in p[1]]
        #if len(sim_proc_list) > 0:
        #    self.program = 'viewer'
        #else:
        self.program = 'runner'

        # Launch sim
        if self.program == 'runner':
            #sim_proc_list = [p[0] for p in processes if 'h3drunner' in p[1]]

            #for p in sim_proc_list:
            #    os.kill(p.pid, signal.SIGTERM)
            path = pjoin(sim_path, 'ISIH3DModuleBase', 'modules', 'NeedleDriving')
            self.sim_pid = ISISim.common_module_starter.startModule(
                'NeedleDriving', 1, 'debug', 'runner', path,
                resolution=self.resolution, random_num=random_num,
                server_num=self.server_num)

            time.sleep(6)

        # Loop until sim is up
        while self.sim_connection is None:
            # Create sim_connection
            self.sim_connection = SimulatorConnection.SimulatorConnection(
                'localhost', self.port, via_daemon=False)

            # Get the simulation version string
            result = self.sim_connection.call(Action('getAPIName'))
            if result.isSuccess():
                apiName = result.getContentText()
            else:
                self.sim_connection = None
                print 'Could not connect to network API.'
                self.error_ctr += 1
                if self.error_ctr > self.max_error_ctr:
                    return False
                time.sleep(6)

        print 'Connected to ' + apiName
        self.error_ctr = 0

        # Create a callback and tell the simulation about it (event)
        callback_id = self.sim_connection.registerSimpleEvent(self._callback)
        result = self.sim_connection.startCallbackListener(
            'localhost', self.env_port)
        if not result.isSuccess():
            print 'ERROR: SimulatorConnection::startCallbackListener' \
                'could NOT be performed successfully.'
            print result.getDescription()
            print result.getContentText()
            sys.exit()

        # Set the callback id in the simulator
        self._callAndCheckAction('setEventId', callback_id)

        self._callAndCheckAction('initSim')

        self._wait_for_sim()

        # Now our connection to the sim is complete
        return True

    def render(self, save_path='./out', sim_save=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Write reward on image
        # pygame is (x,y) while our array is (y,x)
        surface = pygame.surfarray.make_surface(self.image.transpose((1,0,2)))

        if not sim_save:
            n = self.state.needle_tip_pos
            t = self.state.cur_target_pos
            reward_s = 'S{} TR {:.3f}  R {:.3f}  EP {}  d {:.3f}' \
                ' n:({:.2f},{:.2f},{:.2f}) t:({:.2f},{:.2f},{:.2f})'.format(
                self.server_num, self.total_reward, self.last_reward,
                self.episode, self.reward.last_dist,
                n[0], n[1], n[2], t[0], t[1], t[2]
                )
            try:
                txt_surface = self.font.render(reward_s, False, (255,0,0))
            except:
                print "reward_s = ", reward_s
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
            print "convert_to_video: No files found for ", pattern

    def _wait_for_sim(self):
        # Wait for reply: both the packet and the shared mem
        # debug
        #print "rcv_msg_id is {}, ready_flag is {}".format(
        #    self.rcv_msg_id, self.shared_vars[0])

        while (self.rcv_msg_id < self.msg_id or
               self.ready_flag_old == self.shared_vars[0]):
            # debug
            #print "rcv_msg_id is {}, ready_flag is {}".format(
            #    self.rcv_msg_id, self.shared_vars[0])
            time.sleep(0.04)

    def _update_sim_state(self, action=None):

        self._wait_for_sim()

        image = self._read_shared_data()

        # Resize/crop image
        if self.stereo_mode:
            x = [0., 1.]
        else:
            x = [0., 0.5] # Get half the image (left one)
        y = [0.40, 0.82] # Get relevant part of image
        cropx = [int(self.resolution[0] * x[0]), int(self.resolution[0] * x[1]) + 1]
        cropy = [int(self.resolution[1] * y[0]), int(self.resolution[1] * y[1]) + 1]
        image = image[cropy[0]:cropy[1], cropx[0]:cropx[1], :]

        w, h = self._get_width_height(hires=True, stereo=True)
        self.image = scipy.misc.imresize(image, (h, w))

        event = self.last_event
        self.last_event = None

        # Backup old state, read all the state in the event
        self.last_state = self.state
        self.state = State()

        self.state.from_data(event)
        self.state.action = action
        #print "_update_sim_state: action = ", self.state.action # debug

    def _get_env_state(self):
        image = self.image
        # Resize to non-hires, but possibly stereo
        w, h = self._get_width_height(hires=False, stereo=True)
        if self.hi_res_mode:
            image = scipy.misc.imresize(self.image, (h, w))
        if self.stereo_mode:
            # h = w / 2
            image_l = image[:, :h, :]
            image_r = image[:, h:, :]
            image = np.concatenate([image_l, image_r], axis=2)

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

        if self.sim_connection is None:
            self._connect_to_sim()
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
            # Suture expects a 7-member action (1st is dummy)
            arm = 2

            if EXPERT_STRATEGY:
                if self.t>5:
                    pos = [0., 0., 0.]
                    orn = [0., 0., -0.15]
                else:
                    pos = [0., -0.001, 0.]
                    orn = [0., 0., 0.]
            else:
                pos = [a[1]*0.001, a[2]*0.001, a[3]*0.001]
                orn = [0., 0., a[4]*0.1]
        else:
            raise ValueError("Invalid task " + str(self.task))

        action_l = [arm, pos, orn, 0] # last is jaw

        self._callAndCheckAction('moveArm', action_l)

        self._update_sim_state(action=a_orig)

    # Action space:
    # one arm only, Position (3), orientation (3)
    def step(self, action_orig):

        if self.done:
            print "[{}] In step: Need reset".format(self.server_num)
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
        reward, done = self.reward.get_reward_and_done()
        done = done or play_done

        self.done = done
        self.last_reward = reward
        self.total_reward += reward

        # Save the img/state action of every step
        # Upon reset, we'll dump the states file
        self._step_record()

        return (cur_state, reward, done,
            {"action": action_orig, "save_mode":self.get_save_mode()})

    def _reset_real(self):
        if self.sim_connection is None:
            self._connect_to_sim()

        if self.episode % self.reboot_eps == 0 or \
           self.error_ctr >= self.max_error_ctr:
            self._reboot()

        self.error_ctr = 0

        # Loop until we get a proper reset
        good = False
        while not good:
            if self.error_ctr > 5:
                self._reboot()
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

    def _get_width_height(self, hires=False, stereo=False):
        img_dim = self.img_dim * 2 if self.hi_res_mode and hires else self.img_dim
        w = img_dim * 2 if self.stereo_mode and stereo else img_dim
        h = img_dim
        return w, h

    def reset(self, sim_save=True, **kwargs):
        self.t = 0
        self.episode += 1
        self.done = False
        self.insert_status = 0
        self.total_reward = 0.
        self.last_reward = 0.
        self.last_target = None

        w, h = self._get_width_height(hires=True, stereo=True)
        if not self._reset_try_play(w, h):
            self._reset_real()

        self._reset_record()

        cur_state = self._get_env_state()

        # Things required for the reward function
        self.reward.reset()

        return (cur_state, 0, False, {"action": None, "save_mode": self.get_save_mode()})

    @staticmethod
    def clean_up_env():
        # Check if a sim is running
        processes = psutil.process_iter()
        sim_proc_list = []
        for p in processes:
            if 'h3drunner' in p.name().lower() and \
                p.username() == getpass.getuser():
                  sim_proc_list.append(p)

        for p in sim_proc_list:
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
            print "here"
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
           print "Commands: r(eset), s(ave), l(oad), 1/2/3"

