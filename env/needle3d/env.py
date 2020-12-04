# -*- coding: utf-8 -*-
import os, math, random, copy
from os.path import join as pjoin
import weakref

import numpy as np
import shapely.geometry as geo
import pygame as pg
from PIL import Image

from env.common_env import CommonEnv
from . import graphics as gr

import cProfile, pstats

GREEN = np.array([0., 255., 0., 255.],dtype=np.float32) / 255.
LIGHT_BLUE = np.array([33., 65., 243., 255.],dtype=np.float32) / 255.
ORANGE = np.array([214., 139., 19., 255.],dtype=np.float32) / 255.
PINK = np.array([245., 66., 236., 255.],dtype=np.float32) / 255.
GREENISH = np.array([100., 175., 100., 255.], dtype=np.float32) / 255.
REDDISH = np.array([175., 100., 100., 255.], dtype=np.float32) / 255.

'''
For best results, use pillow-simd:

$ pip uninstall pillow
$ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
'''

two_pi = math.pi * 2
pi_div2 = math.pi / 2

VELOCITY = 50

def safe_load_line(name, handle):
    l = handle.readline()[:-1].split(': ')
    assert(l[0] == name)

    return l[1].split(',')

def rgb2gray(rgb):
    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype(np.uint8)
    gray = np.expand_dims(gray, 0)
    return gray

class State:
    pass

PROFILE=False

if PROFILE:
    g_pr = cProfile.Profile()

def dump_stats():
    ps = pstats.Stats(g_pr)
    ps.sort_stats('calls', 'cumtime')
    ps.print_stats()

class Environment(CommonEnv):
    metadata = {
            'render.modes': ['image', 'state'],
            'camera.modes' : ['ortho', 'topdown', 'bottom', 'right'],
            'object.modes' : ['2d', '3d' ]
            }
    background_color = np.array([99., 153., 174., 255.]) / 255.

    def __init__(self, mode='image', stack_size=1,
            log_file=None, filename=None, max_steps=150, img_dim=224,
            action_steps=51,
            random_env=True, random_needle=True,
            min_gates=3, max_gates=3,
            scale_rewards=False,
            add_delay=0.,
            full_init=True,
            camera='ortho',
            object_mode='2d',
            **kwargs):

        super(Environment, self).__init__(**kwargs)

        self.scale_rewards = scale_rewards # 0 to 1

        # 51 gradations by default
        self.action_steps = np.array([action_steps])
        self.action_dim = len(self.action_steps)

        self.t = 0
        self.max_steps = max_steps
        self.render_mode = mode
        self.camera_mode = camera
        self.object_mode = object_mode
        self.shader_mode = gr.LIGHTING if object_mode == '3d' else gr.DEFAULT
        self.episode = 0
        self.total_time = 0
        self.render_ep_path = None
        self.add_delay = add_delay

        """ create image stack """
        self.stack_size = stack_size
        self.log_file = log_file
        self.img_dim = img_dim
        self.random_env = random_env
        self.random_needle = random_needle
        self.max_gates = max_gates
        self.min_gates = min_gates

        pg.font.init()

        # Set up state
        self.state = State()
        self.state.height = 0
        self.state.width = 0
        self.state.needle = None
        self.state.next_gate = None
        self.state.filename = filename
        self.state.status = None
        self.extra_state_dim = 0

        self.profile_time = 1000

        self.renderer = None

        if full_init:
            self.reset()

    @staticmethod
    def clean_up_env():
        pass

    def combine_states(self, states):
        return np.concatenate(states)

    def sample_action(self):
        action = np.array([random.uniform(-1, -1), random.uniform(1, 1)])
        return action

    def _reset_real(self):

        if self.add_delay > 0.:
            time.sleep(self.add_delay)

        # Modify state
        self.state.ngates = 0
        self.state.gates = []
        self.state.surfaces = []
        self.state.action = None
        # environment damage is the sum of the damage to all surfaces
        self.state.damage = 0
        self.state.next_gate = None

        self.state.width = 1920
        self.state.height = 1080

        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        if self.renderer is None:
            self.renderer = gr.OpenGLRenderer(
                    res=(976,976),
                    #kres=(self.state.width, self.state.height),
                    #bg_color=self.background_color
                    )
            cm = self.camera_mode
            if cm == 'ortho':
                self.renderer.set_ortho(0., float(self.state.width), 0., float(self.state.height))
            elif cm == 'topdown':
                self.renderer.set_perspective()
                self.renderer.set_camera_loc((self.state.width / 2., self.state.height / 2., 2500.))
                self.renderer.set_camera_lookat((self.state.width/2., self.state.height/2., 0.))
                self.renderer.set_light_pos((self.state.width/2., self.state.height/2., 400.))
                self.renderer.update_view_matrix()
            else:
                self.renderer.set_perspective()
                self.renderer.set_camera_up((0., 0., 1.))
                if cm == 'bottom':
                    loc = (self.state.width/2., -300., 800.)
                    lookat = (self.state.width/2., self.state.height/2., -100.)
                elif cm == 'right':
                    loc = (self.state.width + 300, self.state.height/2., 800.)
                    lookat = (self.state.width/2., self.state.height/2., -100.)
                elif cm  == 'left':
                    loc = (-500, self.state.height/2., 1900.)
                    lookat = (self.state.width/2. - 100., self.state.height/2., -100.)
                elif cm == 'lowleft':
                    loc = (-500, self.state.height/2., 10.)
                    lookat = (self.state.width/2. - 100., float(self.state.height), 10.)
                self.renderer.set_camera_loc(loc)
                self.renderer.set_camera_lookat(lookat)
                self.renderer.set_light_pos((self.state.width/2., self.state.height/2., 400.))
                self.renderer.update_view_matrix()

        # Init env
        if self.random_env:
            self.create_random_env()
        elif self.state.filename is not None:
            with open(self.state.filename, 'r') as file:
                self.load(file)
        else:
            raise ValueError('No file to run')

        self.state.needle = Needle(self, self.renderer, self.state.width, self.state.height,
            self.log_file, random_pos=self.random_needle)

        self.image = self._draw()

    def reset(self, render_ep_path=None):
        ''' Create a new environment. Currently based on attached filename '''

        # Convert last episode if needed
        if self.render_ep_path is not None:
            self.convert_to_video(self.render_ep_path, sim_save=False)

        self.render_ep_path=render_ep_path
        self.t = 0
        self.episode += 1
        self.done = False
        self.total_reward = 0.
        self.last_reward = 0.
        self.last_dist = None

        if not self._reset_try_play(self.img_dim, self.img_dim):
            self._reset_real()

        self.stack = [self.image] * self.stack_size

        self._reset_record()

        # Get image stack
        stack = [x.transpose((2,0,1)) for x in self.stack]
        ob = np.array(stack)

        # Handle episode render if needed
        if self.render_ep_path is not None:
            self.render(self.render_ep_path, sim_save=False)

        if self.render_mode in ['state', 'mixed']:
            st = self._get_env_state().reshape((1,-1))

        if self.render_mode == 'image':
            cur_state = ob
        elif self.render_mode == 'state':
            cur_state = st
        elif self.render_mode == 'mixed':
            cur_state = (ob, st)

        extra = {"action":None, "save_mode":self.get_save_mode(), "success":False}
        return (cur_state, 0, False, extra)

    def _draw(self):
        self.renderer.start_draw()

        #for surface in self.state.surfaces:
            #surface.draw()

        self.background.draw()

        for gate in self.state.gates:
            gate.draw()

        self.state.needle.draw()

        # DEBUG
        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        pixels = self.renderer.get_img()
        frame = Image.fromarray(pixels)
        frame = frame.resize((self.img_dim, self.img_dim), resample=Image.NEAREST)
        frame = np.asarray(frame)
        return frame

    def render(self, save_path='./out', sim_save=False):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        surface = pg.surfarray.make_surface(self.image.transpose((1,0,2)))

        # draw text
        if not sim_save:
            myfont = pg.font.SysFont('Arial', 10)
            '''
            reward_s = "S:{} TR:{:.3f}, R:{:.3f} a:{}".format(
                self.server_num, self.total_reward, self.last_reward,
                self.state.action)
            '''
            needle = self.state.needle
            reward_s = 'x:{:.3f}, y:{:.3f}, w:{:.3f}'.format(
                    float(needle.x), float(needle.y), float(needle.w))
            txt_surface = myfont.render(reward_s, False, (0, 0, 0))
            surface.blit(txt_surface, (10, 10))

        save_name = self._gen_save_name(sim_save=sim_save) + '.png'
        save_file = pjoin(save_path, save_name)
        pg.image.save(surface, save_file)


    @staticmethod
    def parse_name(filename):
        toks = filename.split('/')[-1].split('.')[0].split('_')
        return toks[1]

    '''
    Load an environment file.
    '''
    def load(self, handle):

        D = safe_load_line('Dimensions', handle)
        self.state.height = int(D[1])
        self.state.width = int(D[0])
        #print(" - width=%d, height=%d"%(self.width, self.height))

        self.create_background()

        D = safe_load_line('Gates', handle)
        self.state.ngates = int(D[0])
        #print(" - num gates=%d"%(self.ngates))

        for _ in range(self.state.ngates):
            gate = Gate(self, self.renderer, self.state.width, self.state.height)
            gate.load(handle, min(self.state.width, self.state.height))
            self.state.gates.append(gate)

        if self.state.ngates > 0:
            self.state.next_gate = 0
            self.state.gates[0].status = 'next'

        if self.max_gates < self.state.ngates:
            self.max_gates = self.state.ngates

        D = safe_load_line('Surfaces', handle)
        self.state.nsurfaces = int(D[0])

        for i in range(self.state.nsurfaces):
            s = Surface(self.state.width, self.state.height)
            s.load(handle)
            self.state.surfaces.append(s)

    def create_background(self):
        self.background = self.renderer.create_rectangle(self.shader_mode)
        self.background.set_color(self.background_color)
        self.background.translate((self.state.width/2., self.state.height/2., 0.))
        self.background.scale((self.state.width * 1.5, self.state.height * 1.5, 1.))

    def create_random_env(self):
        self.state.width = 1920
        self.state.height = 1080

        self.create_background()

        self.state.ngates = random.randint(self.min_gates, self.max_gates)

        # Create gates that don't overlap with any others
        for _ in range(self.state.ngates):
            gate = Gate(self, self.renderer, self.state.width, self.state.height)
            overlaps = True
            while overlaps:
                rand = np.random.rand(3)
                gate.from_params(*rand)
                overlaps = False
                for gate2 in self.state.gates:
                    if gate.box.intersects(gate2.box):
                        overlaps = True
            self.state.gates.append(gate)

        if self.state.ngates > 0:
            self.state.next_gate = 0
            self.state.gates[0].status = 'next'

        self.state.nsurfaces = 0 # TODO

    def _compute_reward_and_done(self):
        reward = 0
        done = False

        status = self.state.gate_status
        if status == 'passed':
            self.last_dist = None
            reward += 100
        elif status == 'failed':
            self.last_dist = None
            reward += 20 # Encourage getting there
        elif status == 'done':
            self.last_dist = None
            done = True

        needle = self.state.needle

        # Distance reward component
        next_gate = self._get_next_gate()
        if next_gate is not None:
            x2gate = needle.x - next_gate.x
            y2gate = needle.y - next_gate.y
            dist = np.sqrt(x2gate * x2gate + y2gate * y2gate)
            if self.last_dist is not None:
                delta = (self.last_dist - dist)/1000
                if delta < 0:
                    #delta *= 10.
                    delta = -0.8
                elif delta == 0:
                    delta = -0.5 # no standing still!
                else:
                    pass
                    #delta = 0.05
                reward += delta
            self.last_dist = dist

        # Time penalty
        # Makes circles not rewarding
        if not done:
            reward -= 0.01

        # Check for leaving window
        if (needle.x <= 0 or needle.x >= self.state.width or
            needle.y <= 0 or needle.y >= self.state.height):
            pass
            #reward -= 0.5
            #done = True

        if self._deep_tissue_intersect():
            reward -= 20.
            done = True

        # Damage component
        reward -= self.state.new_damage / 100

        # Check for excessive damage
        if self.state.damage > 100:
            reward -= 20
            done = True

        if self.t > self.max_steps:
            done = True

        if self.scale_rewards:
            # Need to make sure we'll be positive
            reward /= 10
            if reward < 0.:
                reward = 0.
        else:
            reward /= 10

        return reward, done

    def _get_env_state(self):
        ''' Get state in a way the NN can read it '''
        if self.state.next_gate is not None and \
            self.state.next_gate < len(self.state.gates):
            gate = self.state.gates[self.state.next_gate]
            gate_x, gate_y, gate_w = gate.x, gate.y, gate.w
        else:
            gate_x, gate_y, gate_w = 0., 0., 0.
        gate_x /= self.state.width
        gate_y /= self.state.height
        gate_w /= two_pi

        s = []
        s.append(float(self.state.needle.x) / self.state.width)
        s.append(float(self.state.needle.y) / self.state.height)
        s.append(float(self.state.needle.w) / two_pi)
        s.extend([float(gate_x), float(gate_y), float(gate_w)])
        s = np.array(s, dtype=np.float32)
        #print("state = ", s) # debug
        return s

    def _step_real(self, action_orig):

        if self.add_delay > 0.:
            time.sleep(self.add_delay)

        action = np.copy(action_orig)
        action *= 0.25 * math.pi
        self.state.action = action_orig
        needle_surface = self._surface_with_needle()
        self.state.needle.move(action, needle_surface)
        new_damage = self._get_new_damage(action, needle_surface)
        self._update_damage_and_color(needle_surface, new_damage)
        self.state.new_damage = new_damage
        self.state.damage += new_damage
        self._update_next_gate_status()

        self.image = self._draw()

    def step(self, action_orig):
        """
            Move one time step forward
            Returns:
              * state of the world (in our case, an image)
              * reward
              * done
        """

        if PROFILE:
            if self.total_time % self.profile_time == 0 and self.total_time > 0:
                dump_stats()
            g_pr.enable()

        if self.done:
            print("[{}] In step: Need reset".format(self.server_num))
            return

        self.t += 1
        self.total_time += 1

        play_done = False
        if self.get_save_mode() == 'play':
            # Dummy playback. Overwrite action
            action_orig, play_done = self._step_play()
        else:
            self._step_real(action_orig)

        self.stack.pop(0)
        self.stack.append(self.image)
        assert len(self.stack) == self.stack_size

        reward, done = self._compute_reward_and_done()
        #print("reward: {} done: {}".format(reward, done)) # debug

        done = done or play_done

        self.done = done
        self.last_reward = reward
        self.total_reward += reward

        stack = [x.transpose((2,0,1)) for x in self.stack]
        ob = np.array(stack)

        if self.render_mode in ['state', 'mixed']:
            """ else from state to action"""
            cur_state = self._get_env_state().reshape((1,-1))

        self._step_record()

        # Handle epsiode render if needed
        if self.render_ep_path is not None:
            self.render(self.render_ep_path, sim_save=False)

        st = None
        if self.render_mode == 'image':
            st = ob
        elif self.render_mode == 'state':
            st = cur_state
        elif self.render_mode == 'mixed':
            st = (ob, cur_state)

        extra = {
            "action": action_orig,
            "save_mode": self.get_save_mode(),
            "success": self._get_next_gate_status() == 'done',
            "best_action": None,
            "extra_state": None
        }

        if PROFILE:
            g_pr.disable()

        return (st, reward, done, extra)

    def _surface_with_needle(self):
        for s in self.state.surfaces:
            if self._needle_in_surface(s):
                return s
        return None

    def _get_new_damage(self, movement, surface):
        if surface is not None:
            return surface.get_new_damage(movement)
        return 0.

    def _update_damage_and_color(self, surface, new_damage):
        if surface is not None:
            surface.update_damage_and_color(new_damage)

    def _needle_in_surface(self, s):
        ''' @s: a surface '''
        return s.contains(self.state.needle.tip)

    def _get_next_gate_status(self):
        next_gate = self._get_next_gate()
        if next_gate is None:
            return 'done'
        return next_gate.status

    def _get_next_gate(self):
        if self.state.next_gate is None or \
           self.state.next_gate >= len(self.state.gates):
            return None
        return self.state.gates[self.state.next_gate]

    def _update_next_gate_status(self):
        """ verify if the game is in a valid state and can
            keep playing """
        # have we passed a new gate?
        next_gate = self._get_next_gate()
        if next_gate is None:
            self.state.gate_status = 'done'
            return

        next_gate._update_status_and_color(
            self.state.needle.tip, self.state.needle.last_tip)
        status = self._get_next_gate_status()
        self.state.gate_status = status
        # if you passed or failed the gate
        if status in ['failed', 'passed']:
            # increment to the next gate
            self.state.next_gate += 1
            next_gate = self._get_next_gate()
            if next_gate is not None:
                next_gate.status = 'next'

    def _deep_tissue_intersect(self):
        """
            check each surface, does the needle intersect the
            surface? is the surface deep?
        """
        for s in self.state.surfaces:
            if s.deep and self._needle_in_surface(s):
                return True
        return False

class Gate:
    color_passed = GREENISH
    color_failed = REDDISH
    color_next = PINK
    color1 = np.array([251., 216., 114., 255.]) / 255.
    color2 = np.array([255., 50., 12., 255.]) / 255.

    def __init__(self, env, renderer, env_width, env_height):

        self.env = env
        self.renderer = renderer

        self.x = 0.
        self.y = 0.
        self.w = 0.
        self.top = np.zeros((4,2))
        self.bottom = np.zeros((4,2))
        self.corners = np.zeros((4,2))
        self.width = 0
        self.height = 0
        self.status = None
        self.c_mid = self.color1
        self.c_outer = self.color2

        self.box = None
        self.bottom_box = None
        self.top_box = None

        self.env_width = env_width
        self.env_height = env_height

    def _update_status_and_color(self, p, last_p):
        ''' take in current position,
            see if you passed or failed the gate
        '''
        path = geo.LineString([last_p, p])

        if self.status != 'passed' and \
                (path.intersects(self.top_box) or
                path.intersects(self.bottom_box)):
            self.status = 'failed'
            self.c_mid = self.color_failed
            self.c_outer = self.color_failed
        elif self.status == 'next':
            if self.box.contains(p):
                self.status = 'passed'
                self.c_mid = self.color_passed
                self.c_outer = self.color_passed
            else:
                self.c_mid = self.color_next

    def draw(self):
        self.top_obj.set_color(self.c_outer)
        self.top_obj.draw()
        self.mid_obj.set_color(self.c_mid)
        self.mid_obj.draw()
        self.bot_obj.set_color(self.c_outer)
        self.bot_obj.draw()

    def from_params(self, x, y, w, length=None, width=None):
        ''' @x, y, w: -3.14 to 0 to 3.14
            y is 0 at bottom, goes up
            x is 0 at left
            w is 0 at left, pos going cw
        '''
        scale = min(self.env_height, self.env_width)

        gl = 0.25 if length is None else length
        gw = gl / 3. if width is None else width
        gh = gw / 2 # height if needed
        bl = gl / 10.

        h_gw = gw / 2.
        h_gl = gl / 2.
        h_bl = bl / 2.
        sinw = math.sin(w)
        cosw = math.cos(w)

        # order of corners: TR, BR, BL, TL
        # Remember y starts from below
        self.corners = np.array([
           [x + h_gl * cosw - h_gw * sinw,
            y + h_gl * sinw + h_gw * cosw],
           [x + h_gl * cosw + h_gw * sinw,
            y + h_gl * sinw - h_gw * cosw],
           [x - h_gl * cosw + h_gw * sinw,
            y - h_gl * sinw - h_gw * cosw],
           [x - h_gl * cosw - h_gw * sinw,
            y - h_gl * sinw + h_gw * cosw],
           ])
        self.top = np.array(self.corners)
        self.top[3,:] = self.corners[0, :]
        self.top[2,:] = self.corners[1, :]
        self.top[3,0] -= h_bl * cosw
        self.top[3,1] -= h_bl * sinw
        self.top[2,0] -= h_bl * cosw
        self.top[2,1] -= h_bl * sinw
        self.bottom = np.array(self.corners)
        self.bottom[1,:] = self.corners[2, :]
        self.bottom[0,:] = self.corners[3, :]
        self.bottom[1,0] += h_bl * cosw
        self.bottom[1,1] += h_bl * sinw
        self.bottom[0,0] += h_bl * cosw
        self.bottom[0,1] += h_bl * sinw

        self.x = x * self.env_width
        self.y = y * self.env_height
        self.w = w
        #print("corners: ", self.corners) # debug
        self.corners[:,0] *= self.env_width
        self.corners[:,1] *= self.env_height
        self.top[:,0] *= self.env_width
        self.top[:,1] *= self.env_height
        self.bottom[:,0] *= self.env_width
        self.bottom[:,1] *= self.env_height

        # Graphics
        if self.env.object_mode =='2d':
            create_fun = self.renderer.create_rectangle
        elif self.env.object_mode =='3d':
            create_fun = self.renderer.create_cube
        else:
            raise ValueError("Unknown object_mode " + self.env.object_mode)
        self.mid_obj = create_fun(shader=self.env.shader_mode)
        self.top_obj = create_fun(shader=self.env.shader_mode)
        self.bot_obj = create_fun(shader=self.env.shader_mode)

        self.mid_obj.translate((self.x, self.y, 0.))
        self.top_obj.translate((self.x, self.y, 0.))
        self.bot_obj.translate((self.x, self.y, 0.))

        self.mid_obj.rotate(w + pi_div2)
        self.top_obj.rotate(w + pi_div2)
        self.bot_obj.rotate(w + pi_div2)

        self.top_obj.translate((0., h_gl * scale, 0.))
        self.bot_obj.translate((0., -h_gl * scale, 0.))

        z_scale = gh * scale if self.env.object_mode == '3d' else 1.
        self.mid_obj.scale((gw * scale, (gl - bl) * scale, z_scale))
        self.top_obj.scale((gw * scale, bl * scale, z_scale))
        self.bot_obj.scale((gw * scale, bl * scale, z_scale))

        self.mid_obj.translate((0., 0., 0.5))
        self.top_obj.translate((0., 0., 0.5))
        self.bot_obj.translate((0., 0., 0.5))

        #print("corners2: ", self.corners) # debug

        self._create_polys()


    '''
    Load Gate from file at the current position.
    '''
    def load(self, handle, scale):

        pos = safe_load_line('GatePos', handle)
        cornersx = safe_load_line('GateX', handle)
        cornersx = np.array([float(x) for x in cornersx])
        cornersy = safe_load_line('GateY', handle)
        cornersy = np.array([float(x) for x in cornersy])
        topx = safe_load_line('TopX', handle)
        topy = safe_load_line('TopY', handle)
        bottomx = safe_load_line('BottomX', handle)
        bottomy = safe_load_line('BottomY', handle)

        #from rl.utils import ForkablePdb
        #ForkablePdb().set_trace()

        diffx = np.array([cornersx[1] - cornersx[0], cornersx[2] - cornersx[1]])
        diffy = np.array([cornersy[1] - cornersy[0], cornersy[2] - cornersy[1]])
        diffs = np.sqrt(diffx * diffx + diffy * diffy)
        length = np.max(diffs)
        width = np.min(diffs)

        pos = [float(x) for x in pos]
        self.from_params(*pos, length=length/scale, width=width/scale)

        '''
        self.x = self.env_width * float(pos[0])
        self.y = self.env_height * float(pos[1])
        self.w = float(pos[2])

        self.top[:, 0] = [float(x) for x in topx]
        self.top[:, 1] = [float(y) for y in topy]
        self.bottom[:, 0] = [float(x) for x in bottomx]
        self.bottom[:, 1] = [float(y) for y in bottomy]
        self.corners[:, 0] = [float(x) for x in cornersx]
        self.corners[:, 1] = [float(y) for y in cornersy]

        # apply corrections to make sure the gates are oriented right
        self.w *= -1
        if self.w < 0:
            self.w = self.w + (np.pi * 2)
        if self.w > np.pi:
            self.w -= np.pi
            self.top = np.squeeze(self.top[np.ix_([2, 3, 0, 1]), :2])
            self.bottom = np.squeeze(self.bottom[np.ix_([2, 3, 0, 1]), :2])
            self.corners = np.squeeze(self.corners[np.ix_([2, 3, 0, 1]), :2])

        self.w -= np.pi / 2

        self._create_polys()
        '''

    def _create_polys(self):

        avgtopy = np.mean(self.top[:, 1])
        avgbottomy = np.mean(self.bottom[:, 1])

        # flip top and bottom if necessary
        if avgtopy < avgbottomy:
            tmp = self.top
            self.top = self.bottom
            self.bottom = tmp

        # compute other things like polygon
        self.box = geo.Polygon(self.corners)
        self.top_box = geo.Polygon(self.top)
        self.bottom_box = geo.Polygon(self.bottom)

class Surface:

    def __init__(self, env_width, env_height):
        self.deep = False
        self.corners = None
        self.color = None
        self.damage = 0 # the damage to this surface

        self.env_width = env_width
        self.env_height = env_height

        self.poly = None

    def contains(self, x):
        self.poly.contains(x)

    def draw(self, surface):
        ''' update damage and surface color '''
        #pg.draw.polygon(surface, self.color, self.corners) # TODO

    '''
    Load surface from file at the current position
    '''
    def load(self, handle):
        isdeep = safe_load_line('IsDeepTissue', handle)

        sx = [float(x) for x in safe_load_line('SurfaceX', handle)]
        sy = [float(x) for x in safe_load_line('SurfaceY', handle)]
        self.corners = np.array([sx, sy]).transpose()
        self.corners[:, 1] = self.env_height - self.corners[:, 1]

        self.deep = isdeep[0] == 'true'
        self.deep_color = np.array([207., 69., 32., 255.]) / 255.
        self.light_color = np.array([232., 146., 124., 255.]) / 255.
        self.color = np.array(self.deep_color if self.deep else self.light_color)

        self.poly = geo.Polygon(self.corners)

    def get_new_damage(self, movement):
        # Check for 2 components
        if len(movement) == 1:
            dw = movement[0]
        else:
            dw = movement[1]
        if abs(dw) > 0.02:
            return (abs(dw) / 2.0 - 0.01) * 100
        return 0.

    def update_damage_and_color(self, new_damage):
        if new_damage > 0:
            self.damage += new_damage
            if self.damage > 100:
                self.damage = 100
            self._update_color()

    def _update_color(self):
        alpha = self.damage / 100.
        beta = (100. - self.damage) / 100.
        self.color = beta * self.light_color + alpha * self.deep_color

class Needle:

    # Assume w=0 points to the negative x-axis

    def __init__(self, env, renderer, env_width, env_height, log_file, random_pos=False):

        self.env = env
        self.renderer = renderer
        if random_pos:
            self.x = random.randint(0, env_width - 1)
            self.y = random.randint(0, env_height - 1)
            self.w = random.random() * two_pi
        else:
            self.x = 96
            self.y = env_height - 108
            self.w = math.pi # face right
        self.dx = 0.0
        self.dy = 0.0
        self.dw = 0.0
        self.corners = None

        self.length_const = 0.12
        self.scale = env_width / 10.
        self.is_moving = False

        self.env_width = env_width
        self.env_height = env_height

        self.needle_color = LIGHT_BLUE
        self.thread_color = np.array([167., 188., 214., 255.]) / 255.

        # Save adjusted thread pointsmath.since we don't use them for anything
        self.thread_points = [(self.x, self.y)]
        self.tip = geo.Point(np.array([self.x, self.y]))
        self.last_tip = self.tip
        self.path_length = 0.

        self.log_file = log_file

        # Graphics

        if self.env.object_mode == '2d':
            create_fun = self.renderer.create_triangle
        elif self.env.object_mode =='3d':
            create_fun = self.renderer.create_pyramid
        self.obj = create_fun(shader=self.env.shader_mode)
        self.obj.set_color(self.needle_color)

        self._load()


    def draw(self):
        #self._draw_thread()
        self._draw_needle()

    def _compute_corners(self):
        """
            given x,y,w compute needle corners and save
        """
        w = self.w
        x = self.x
        y = self.y

        length = self.length_const * self.scale

        lcosw = length * math.cos(w)
        lsinw = length * math.sin(w)
        scale = 0.03 * self.scale

        # Back of the needle
        top_w = w - pi_div2
        bot_w = w + pi_div2

        top_x = x - scale * math.cos(top_w) + lcosw
        top_y = y - scale * math.sin(top_w) + lsinw
        bot_x = x - scale * math.cos(bot_w) + lcosw
        bot_y = y - scale * math.sin(bot_w) + lsinw

        self.corners = np.array([[x, y], [top_x, top_y], [bot_x, bot_y]])

    def _draw_needle(self):
        old_model = self.obj.model
        #self.obj.translate((self.env_width/4., self.env_height/2., 0.))
        self.obj.translate((self.x, self.y, 0.))
        if self.env.object_mode == '3d':
            self.obj.rotate(float(-self.w) - math.pi, (0., 0., 1.))
            self.obj.scale((self.scale, self.scale,  self.scale))
            self.obj.translate((0., 0., 0.5)) # lift
            self.obj.rotate(pi_div2, vec=(0., 1., 0.))
            self.obj.scale((0.5, 0.5, 0.8)) # smaller arrow
        elif self.env.object_mode == '2d':
            self.obj.rotate(float(self.w + pi_div2))
        #self.obj.scale((self.scale * 0.7, self.scale * 1.5, self.scale))
        self.obj.draw()
        self.obj.model = old_model

    def _draw_thread(self, surface):
        if len(self.thread_points) > 1:
            pg.draw.lines(surface, self.thread_color, False, self.thread_points, 10)

    def _load(self):
        """
            Load the current needle position
        """
        # compute the corners for the current position
        self._compute_corners()

    def _action2motion(self, action):
        """ API for reinforcement learning, mapping action to real motion
           action = [main engine, up-down engine]
           needle is set to moving from left to right (constant value)
           main engine: control the moving speed of the needle, -1 ~ 0 : off, 0 ~ 1 : on
           up-down engine: control the moving direction: -1 ~ -0.5: up, 0.5 ~ 1: down, -0.5 ~ 0.5: off
           action is clipped to [-1, 1]

           motion = [dX, dw], dX linear velocity should always be +, dw angular velocity
        """
        w = self.w

        """ 1 dimension action """
        dw = action
        dw = math.pi if dw > math.pi else -math.pi if dw < -math.pi else dw
        dx = math.cos(math.pi - w - dw) * VELOCITY
        dy = -math.sin(math.pi - w - dw) * VELOCITY

        if self.log_file:
            self.log_file.write('action:{}\n'.format(action[0]))
            self.log_file.write('dx:{}, dy:{}, dw:{}\n'.format(dx, dy, dw))
            self.log_file.flush()

        return dw, dx, dy

    def move(self, action, needle_in_tissue):
        """
            Given an input, move the needle. Update the position, orientation,
            and thread path in android game movement is specified by touch
            points. last_x, last_y specify the x,y in the previous time step
            and x,y specify the current touch point

        """

        dw, dx, dy = self._action2motion(action)

        if needle_in_tissue:
            dw *= 0.5
            if (abs(dw) > 0.01):
                dw = 0.02 * np.sign(dw)

        self.w += dw
        if self.w > math.pi:
            self.w -= two_pi
        elif self.w < -math.pi:
            self.w += two_pi

        oldx, oldy = self.x, self.y
        self.x += dx
        self.y -= dy

        # Constrain x and y
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x > self.env_width:
            self.x = self.env_width
        if self.y > self.env_height:
            self.y = self.env_height

        if self.x != oldx or self.y != oldy:
            self.thread_points.append((self.x, self.y))
            dlength = math.sqrt(dx * dx + dy * dy)
            self.path_length += dlength

        self.dx, self.dy, self.dw = dx, dy, dw
        self.last_tip = self.tip
        self.tip = geo.Point(np.array([self.x, self.y]))
        self._compute_corners()

