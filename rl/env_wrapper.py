#from multiprocessing import Process, Pipe
import ray

QUIT = 0
STEP = 1
RESET = 2
RENDER = 3
CONVERT_TO_VIDEO = 4
SET_SAVE_MODE = 5

@ray.remote(num_gpus=0.1)
class EnvReceiver:
    ''' Client end '''
    def set_init(self, env_f, args, server_num):
        self.server_num = np.array(server_num)
        self.env = env_f(args, server_num)

    def step(self, data):
        return self.server_num, self.env.step(data)

    def reset(self, render_ep_path):
        return self.server_num, self.env.reset(render_ep_path=render_ep_path)

    def render(self, save_path):
        self.env.render(save_path=save_path)

    def convert_to_video(self, save_path):
        self.env.convert_to_video(save_path=save_path)

    def set_save_mode(self, x):
        self.env.set_save_mode(x)


class EnvWrapper:
    ''' Server end
        Also provides a local caching layer for frequently-accessed data
    '''
    def __init__(self, server_num, create_env_func, args):

        self.server_num = server_num
        self.obj = EnvReceiver.remote()
        self.obj.set_init(create_env_func, args, server_num)

        self.t = 0
        self.episode = 0
        self.total_reward = 0.
        self.done = False
        self.ready = True
        self.save_mode = ''

    def step(self, action):
        ''' Non-blocking '''
        assert(self.done == False)
        self.obj.step(action)
        self.ready = False
        self.t += 1

    def reset(self, render_ep_path=None):
        ''' Non-blocking '''
        self.done = False
        self.t = 0
        self.total_reward = 0.
        self.episode += 1
        self.actor.reset(render_ep_path)
        self.ready = False

    def reset_block(self):
        self.reset()
        return self.get()

    def step_block(self, action):
        self.step(action)
        return self.get()

    def get(self):
        ''' Blocks to get response '''
        data = ray.get(self.obj)
        [state, reward, done, extra] = data
        self.total_reward += reward
        self.done = done
        self.ready = True
        self.save_mode = extra["save_mode"]
        self.success = extra["success"]
        return (state, reward, done, extra)

    def is_ready(self):
        return self.ready

    def render(self, save_path):
        self.obj.render(save_path)

    def convert_to_video(self, save_path):
        self.obj.convert_to_video(save_path)

    def set_save_mode(self, value):
        self.last_save_mode = self.save_mode
        self.obj.set_save_mode(value)

    def restore_last_save_mode(self):
        last = self.last_save_mode
        self.last_save_mode = self.save_mode
        self.obj.set_save_mode(last)

    def get_save_mode(self):
        return self.save_mode

class EnvWatcher(object):
    def __init__(self, envs):
        self.envs = envs

    def wait(self):
        return ray.wait
