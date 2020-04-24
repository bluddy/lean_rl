from multiprocessing import Process, Pipe

QUIT = 0
STEP = 1
RESET = 2
RENDER = 3
CONVERT_TO_VIDEO = 4
SET_SAVE_MODE = 5

class EnvReceiver:
    ''' Client end '''
    def __init__(self, env, pipe):
        self.env = env
        self.pipe = pipe

    def run(self):
        running = True
        while running:
            v = self.pipe.recv()

            if v[0] == STEP:
                data = self.env.step(v[1])
                self.pipe.send(data)

            elif v[0] == RESET:
                data = self.env.reset(render_ep_path=v[1])
                self.pipe.send(data)

            elif v[0] == QUIT:
                running = False

            elif v[0] == RENDER:
                self.env.render(save_path=v[1])

            elif v[0] == CONVERT_TO_VIDEO:
                self.env.convert_to_video(save_path=v[1])

            elif v[0] == SET_SAVE_MODE:
                self.env.set_save_mode(v[1])

def proc_run_receiver(create_env_func, server_num, args, pipe):
    ''' Run from a new process '''
    env = create_env_func(args, server_num)
    receiver = EnvReceiver(env, pipe)
    receiver.run()

class EnvWrapper:
    ''' Server end '''
    def __init__(self, server_num, create_env_func, args):

        self.server_num = server_num
        pipes = Pipe()
        self.p = Process(target=proc_run_receiver,
                    args=(create_env_func, server_num, args, pipes[1]))
        self.p.daemon = True
        self.pipe = pipes[0]
        self.p.start()

        self.t = 0
        self.episode = 0
        self.total_reward = 0.
        self.done = False
        self.ready = True
        self.save_mode = ''

    def step(self, action):
        ''' Non-blocking '''
        assert(self.done == False)
        self.pipe.send((STEP, action))
        self.ready = False
        self.t += 1

    def reset(self, render_ep_path=None):
        self.done = False
        self.t = 0
        self.total_reward = 0.
        self.episode += 1
        self.pipe.send((RESET, render_ep_path))
        self.ready = False

    def reset_block(self):
        self.reset()
        return self.get()

    def step_block(self, action):
        self.step(action)
        return self.get()

    def get(self):
        ''' Blocks to get response '''
        data = self.pipe.recv()
        [state, reward, done, extra] = data
        self.total_reward += reward
        self.done = done
        self.ready = True
        self.save_mode = extra["save_mode"]
        self.success = extra["success"]
        return (state, reward, done, extra)

    def is_ready(self):
        return self.ready

    def poll(self):
        return self.pipe.poll()

    def render(self, save_path):
        self.pipe.send((RENDER, save_path))

    def convert_to_video(self, save_path):
        self.pipe.send((CONVERT_TO_VIDEO, save_path))

    def set_save_mode(self, value):
        self.last_save_mode = self.save_mode
        self.pipe.send((SET_SAVE_MODE, value))

    def restore_last_save_mode(self):
        l_save_mode = self.last_save_mode
        self.last_save_mode = self.save_mode
        self.pipe.send((SET_SAVE_MODE, l_save_mode))

    def get_save_mode(self):
        return self.save_mode

