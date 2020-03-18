import numpy as np
import os
import core
#from gym import error, spaces
#from gym import utils
import seeding
import pygame
import scipy

try:
    import atari_py
except ImportError as e:
    raise ValueError(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))

# neg 0 pos
# x (left, middle, right), y (down, middle, up),  fire (on, off)
action_steps = np.array([3, 3, 2])
action_dim = len(action_steps)

def clean_up_env(args):
    pass

def combine_states(states, mode):
    return np.concatenate(states)

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)
    return ram


class Environment(core.Env): #, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            game='pong',
            mode=None,
            difficulty=None,
            obs_type='image',
            stack_size=4,
            frameskip=(2, 5),
            repeat_action_probability=0.,
            full_action_space=True,
            server_num=0,
            img_dim=64,
            **kwargs):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        #utils.EzPickle.__init__( self, game, mode, difficulty, obs_type,
        #        frameskip, repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game = game
        self.game_path = atari_py.get_game_path(game)
        self.game_mode = mode
        self.game_difficulty = difficulty
        self.server_num = server_num
        self.img_dim = img_dim
        self.stack_size = stack_size
        self.episode = 0

        if not os.path.exists(self.game_path):
            msg = 'You asked for game %s but path %s does not exist'
            raise IOError(msg % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), \
                "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat(
                'repeat_action_probability'.encode('utf-8'),
                repeat_action_probability)

        self.seed()

        self._action_set = (self.ale.getLegalActionSet() if full_action_space
                            else self.ale.getMinimalActionSet())
        #self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width, screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            #self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
            pass
        elif self._obs_type == 'image':
            #self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
            pass
        else:
            raise ValueError('Unrecognized observation type: {}'.format(self._obs_type))

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)

        if self.game_mode is not None:
            modes = self.ale.getAvailableModes()

            assert self.game_mode in modes, (
                "Invalid game mode \"{}\" for game {}.\nAvailable modes are: {}"
            ).format(self.game_mode, self.game, modes)
            self.ale.setMode(self.game_mode)

        if self.game_difficulty is not None:
            difficulties = self.ale.getAvailableDifficulties()

            assert self.game_difficulty in difficulties, (
                "Invalid game difficulty \"{}\" for game {}.\nAvailable difficulties are: {}"
            ).format(self.game_difficulty, self.game, difficulties)
            self.ale.setDifficulty(self.game_difficulty)

        return [seed1, seed2]

    def _multi_to_single_dim(self, a):
        move = 0
        if a[0] == 0.: # x middle
            if a[1] > 0.5:
                move = 2 # up
            elif a[1] == 0.:
                move = 0 # noop
            elif a[1] < -0.5:
                move = 5 # down
        elif a[0] < -0.5: # x left
            if a[1] > 0.5:
                move = 7 # upleft
            elif a[1] == 0.:
                move = 4 # left
            elif a[1] < -0.5:
                move = 9 # downleft
        elif a[0] > -0.5: # x right
            if a[1] > 0.5:
                move = 6 # upright
            elif a[1] == 0.:
                move = 3 # right
            elif a[1] < -0.5:
                move = 8 # downright
        if a[2] > -0.5:
            if move == 0:
                move = 1 # fire
            else:
                move += 8 # fire+move
        return move

    def step(self, a):
        reward = 0.0
        a = self._multi_to_single_dim(a)
        #print "action_set:", self._action_set # debug
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)

        self.t += num_steps
        frame = self._get_obs()
        self.stack.pop(0)
        self.stack.append(frame)
        # Add batch dim
        obs = np.expand_dims(np.concatenate(self.stack), 0)

        #, {"ale.lives": self.ale.lives()} # was appended below
        return obs, reward, self.ale.game_over()

    def _get_image(self):
        img = self.ale.getScreenRGB2()
        img = scipy.misc.imresize(img, (self.img_dim, self.img_dim))
        return img

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image().transpose((2,0,1)) # pytorch format
        return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        self.t = 0
        self.episode += 1
        frame = self._get_obs()
        self.stack = [frame] * self.stack_size
        # Add batch dim
        obs = np.expand_dims(np.concatenate(self.stack), 0)
        return obs

    def render(self, save_image=True, save_path='./out'):
        img = self._get_image()
        surface = pygame.surfarray.make_surface(img)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file = os.path.join(save_path,
            '{:02d}_{:06d}_{:03d}.png'.format(
              self.server_num, self.episode, self.t))
        pygame.image.save(surface, save_file)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
