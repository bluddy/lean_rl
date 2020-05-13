import numpy as np
import torch
import random, math
import os, sys, argparse
import time, datetime
from os.path import abspath
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
import csv
from joblib import Parallel, delayed

# Test whether the CNN can discern different states properly

cur_dir= os.path.dirname(abspath(__file__))

# Append one dir up to path
sys.path.append(abspath(pjoin(cur_dir, '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import utils
from buffers import CNNBuffer
import scipy.misc
from multiprocessing import Process, Pipe
from env_wrapper import EnvWrapper
from policy.learn_action import LearnAction

# Total counts
total_times, total_loss, total_acc = [],[],[]
timestep = 0
states = []
w_s, w_a = [],[]

def process_state(mode, s):
    # Copy as uint8
    if mode == 'image':
        if s.ndim < 4:
            s = np.expand_dims(s, 0)
        s = torch.from_numpy(s).to(device).float() # possibly missing squeeze(1)
        s /= 255.0
    elif mode == 'state':
        s = torch.from_numpy(s).to(device).float()
    elif mode == 'mixed':
        img = s[0]
        if img.ndim < 4:
            img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        s2 = torch.from_numpy(s[1]).to(device).float()
        s = (img, s2)
    else:
        raise ValueError('Unrecognized mode ' + mode)
    return s

def copy_to_dev(action_dim, batch_size, mode, s, a):
    s = process_state(mode, s)
    a = a.reshape((batch_size, action_dim))
    a = torch.LongTensor(u).to(device)
    return s, a

def run(args):

    program_start_t = time.time()

    if args.env == 'needle':
        from env.needle.env import Environment
        env_dir = pjoin(cur_dir, '..', 'env', 'needle', 'data')
        env_file = pjoin(env_dir, 'environment_' + args.task + '.txt')
        env_name = 'rand' if args.random_env else args.task

        env_suffix = '_a' + str(args.action_steps)
        if args.random_needle:
            env_suffix += 'r'

        basename = 'test_{}_{}_{}{}'.format(
            args.env, env_name, args.mode[0:2], env_suffix)

        suffix = ''
        if args.random_env or args.random_needle:
            suffix += '_'
        if args.random_env:
            suffix += 'r'
        if args.random_needle:
            suffix += 'n'

        save_mode_path = os.path.join('saved_data', '{}_{}{}'.format(
            args.env, env_name, suffix))

    elif args.env == 'sim':
        from env.sim_env.env import Environment
        basename = 'test_{}_{}_{}{}'.format(
            args.env, args.mode[0:2], args.task[:3],
            '_rt' if args.random_env else '')

        suffix = '_'
        if args.random_env:
            suffix += 'r'
        # We always save with _hs, but don't always train/test with it
        suffix += 'hs'
        if args.depthmap_mode:
            suffix += 'd'

        save_mode_path = os.path.join('saved_data', '{}_{}{}'.format(
                args.env, args.task, suffix))

    elif args.env == 'atari':
        from env.atari.atari_env import Environment
        env_data_name = '_' + args.game

    else:
        raise ValueError("Unrecognized environment " + args.env)

    # Save mode arguments
    save_mode = 'play'

    now = datetime.datetime.now()
    time_s = now.strftime('%y%m%d_%H%M')
    run_name = 'runs/{}/{}'.format(basename, time_s)
    if args.name:
        run_name += '_' + args.name
    tb_writer = SummaryWriter(run_name)
    logbase = pjoin('logs',basename)
    logdir = pjoin(logbase, time_s)
    if args.name:
        logdir += '_' + args.name

    # Find last dir
    last_dir = None
    if args.load is None:
        if os.path.exists(logbase):
            for d in sorted(os.listdir(logbase), reverse=True):
                model_dir = pjoin(logbase, d, 'model')
                if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
                    last_dir = d
                    break
    else:
        # Use load_last mechanism to load with direct path
        load_path = pjoin(logbase, args.load)
        if os.path.exists(load_path):
            model_dir = pjoin(load_path, 'model')
            if os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0:
                last_dir = args.load
                args.load_last = True


    def make_dirs(args):
        dirs = [pjoin(logdir, s) for s in ['out', 'test', 'model']]
        for p in dirs:
          if not os.path.exists(p):
              os.makedirs(p)
        return dirs

    out_path, test_path, model_path = make_dirs(args)

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
        #torch.backends.cudnn.enabled = False
    else:
        args.device = torch.device('cpu')

    ## environment setup
    log_f = open(pjoin(logdir, 'log.txt'), 'w')

    # CSV file for test resumption
    csv_f = open(pjoin(logdir, 'log.csv'), 'w')
    csv_wr = csv.writer(csv_f, delimiter=',')

    ## dump args to file
    log_f.write(str(args))
    log_f.write('\n')

    """ setting up environment """
    def create_env(args, server_num, dummy_env=False):
        if args.env == 'needle':
            from env.needle.env import Environment
            return Environment(
                filename=env_file,
                mode=args.mode,
                stack_size = args.stack_size,
                img_dim=args.img_dim,
                random_env=args.random_env,
                random_needle=args.random_needle,
                scale_rewards=args.scale_rewards,
                action_steps=args.action_steps,
                server_num=server_num,
                save_mode_path=save_mode_path,
                save_mode=save_mode,
                cnn_test_mode=True,
                )

        elif args.env == 'sim':
            from env.sim_env.env import Environment
            return Environment(
                mode = args.mode,
                stack_size=args.stack_size,
                img_dim=args.img_dim,
                random_target=args.random_env,
                task=args.task,
                hi_res_mode=args.hi_res_mode,
                stereo_mode=args.stereo_mode,
                depthmap_mode=args.depthmap_mode,
                full_init=not dummy_env,
                server_num=server_num,
                save_mode_path=save_mode_path,
                save_mode=save_mode,
                cnn_test_mode=True,
                )

        elif args.env == 'atari':
            from env.atari.atari_env import Environment
            return Environment(
                    game=args.game,
                    server_num=server_num,
                    stack_size=args.stack_size
                    )
        else:
            raise ValueError(args.env + ' is not a recognized environment!')

    # Cleanup leftover envs
    if args.clean:
        Environment.clean_up_env()

    # Dummy env for access to some internal methods
    dummy_env = create_env(args, 0, dummy_env=True)

    # Create environments
    envs = [EnvWrapper(i, create_env, args) for i in range(args.procs)]

    # Delays for resets, which are slow
    sleep_time = 0.2

    state_dim = 0 # not necessary for image

    # Get state dim dynamically from actual state
    if args.mode == 'state':
        state = envs[0].reset_block()[0]
        state_dim = state.shape[-1]
    elif args.mode == 'mixed':
        state = envs[0].reset_block()[0][1]
        state_dim = state.shape[-1]

    action_steps = dummy_env.action_steps
    action_dim = dummy_env.action_dim

    img_depth = args.img_depth
    if args.stereo_mode or args.depthmap_mode:
        img_depth *= 2

    model = LearnAction(state_dim, action_dim, action_steps, args.stack_size,
            args.mode, network=args.network, lr=args.lr, bn=args.batchnorm,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp,
            deep=args.deep, dropout=args.dropout)

    # Load from files if requested
    if args.load_last and last_dir is not None:
        last_model_dir = pjoin(logbase, last_dir, 'model')
        if args.load_best:
            last_model_dir = pjoin(last_model_dir, 'best')
        timestep_file = pjoin(last_model_dir, 'timestep.txt')
        timestep = None
        if os.path.exists(timestep_file):
            with open(timestep_file, 'r') as f:
                timestep = int(f.read()) + 1
        model.load(pjoin(last_model_dir, 'model.pth'))
        last_csv_file = pjoin(logbase, last_dir, 'log.csv')
        with open(last_csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            t = 0
            for line in reader:
                t = int(line[0])
                # Stop if we know where to stop reading csv
                if timestep is not None and t > timestep:
                    break
                r, q_avg, q_max, loss, best_avg_reward = map(
                    lambda x: float(x), line[1:6])
                last_learn_t, last_eval_t = int(line[6]), int(line[7])
                total_times.append(t)
                total_loss.append(loss)
                csv_wr.writerow(line)
            csv_f.flush()
            if timestep is None:
                timestep = t + 1
        print 'last_model_dir is {}, t={}'.format(last_model_dir, timestep)

    replay_buffers = [CNNBuffer(args.mode, args.capacity, compressed=args.compressed) for _ in range(2)]


    proc_std = []

    # Reset all envs and get first state
    for env in envs:
        env.reset()

    global states
    states = [env.get()[0] for env in envs] # block

    fill_steps = args.capacity / 2

    for _ in xrange(args.epochs):

        for i, replay_buffer in enumerate(replay_buffers):
            global timestep, states, w_s, w_a

            elapsed_time = 0.

            # Fill the replay buffer
            print "\nFilling replay buffer {} with {} steps".format(i, fill_steps)
            start_timestep = timestep
            while timestep - start_timestep < fill_steps:

                acted = False
                start_t = time.time()
                new_states = []

                dummy_action = np.zeros((action_dim,))

                # Send non-blocking dummy action on ready envs
                for env in envs:
                    if env.is_ready():
                        env.step(dummy_action)

                # Save our data so we can loop and insert it into the replay buffer
                for env, state in zip(envs, states):
                    if env.is_ready() or env.poll():
                        # Get the state and saved action from the env
                        new_state, _, done, dict = env.get() # blocking
                        action = dict["action"]
                        new_states.append(new_state)

                        if action is not None: # reset
                            w_s.append(state)
                            w_a.append(dict["best_action"])

                        if done:
                            env.reset(render_ep_path=None) # Send async action

                        acted = True
                        timestep += 1
                    else:
                        # Nothing to do with the env yet
                        new_states.append(state)

                if len(w_s) > 10:
                    # Do compression in parallel
                    if args.compressed:
                        w_s = Parallel(n_jobs=-1)(delayed(state_compress)
                                (args.mode, s) for s in w_s)

                    # Feed into the replay buffer
                    for s, a in zip(w_s, w_a):
                        replay_buffer.add([s, a])
                    w_s, w_a = [],[]

                elapsed_time += time.time() - start_t
                if acted and timestep % 100 == 0:
                    #print "Time: ", elapsed_time # debug
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    elapsed_time = 0.

                states = new_states

        save_mode_playing_cnt = 0
        save_mode_recording_cnt = 0
        for env in envs:
            if env.get_save_mode() == 'play':
                save_mode_playing_cnt += 1
            if env.get_save_mode() == 'record':
                save_mode_recording_cnt += 1

        # Train over collected data

        model.set_train()

        # Train
        temp_loss = []
        for _ in xrange(args.train_loops):
            # Get data from replay buffer
            loss = model.train(replay_buffers[0], args)

            total_loss.append(loss)

            s = '\nTraining T:{} TS:{:04d} L:{:.5f} Exp_std:{:.2f} p:{} r:{}'.format(
                str(datetime.timedelta(seconds=time.time() - program_start_t)),
                timestep,
                loss,
                0 if len(proc_std) == 0 else sum(proc_std)/len(proc_std),
                save_mode_playing_cnt,
                save_mode_recording_cnt
                )
            print s
            proc_std = []

            log_f.write(s + '\n')

        model.set_eval()

        #total_loss.append(np.mean(temp_loss))

        fig = plt.figure()
        plt.plot(total_loss, label='Loss')
        plt.savefig(pjoin(logdir, 'loss.png'))
        tb_writer.add_figure('loss', fig, global_step=timestep)

        # Evaluate
        print('\n---------------------------------------')
        print 'Evaluating CNN for ', logdir
        correct, total = 0, 0
        for _ in xrange(args.eval_loops):

            action, predicted_action = policy.test(replay_buffers[1], args)
            print action, predicted_action, '\n'
            correct += (action == predicted_action).sum()
            total += len(action)

        acc = correct / float(total)

        s = "Eval Accuracy: {:.3f}".format(acc)
        print s
        log_f.write(s + '\n')

        total_acc.append(acc)

        fig = plt.figure()
        plt.plot(total_acc, label='Accuracy')
        plt.savefig(pjoin(logdir, 'acc.png'))
        tb_writer.add_figure('acc', fig, global_step=timestep)

    csv_f.close()
    log_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int,
        help='Number of epochs to train')
    parser.add_argument('--train-loops', default=100, type=int,
        help='How many times to train over data in replay buffer')
    parser.add_argument('--eval-loops', default=100, type=int,
        help='How many times to test over data in replay buffer')
    parser.add_argument('--disable-cuda', default=False, action='store_true',
        help='Disable CUDA')
    parser.add_argument("--env", default="needle",
        help='Environment name [needle|sim|atari]')

    parser.add_argument("--game", default='pong',
        help='Game for atari environment')

    parser.add_argument("--random-needle", default = False, action='store_true',
        dest='random_needle',
        help="Choose whether the needle should be random at each iteration")
    parser.add_argument('--action-steps', default = 51, type=int,
        help="How many action steps to use for needle")
    parser.add_argument("--random-env", default=False, action='store_true',
        dest='random_env', help='Whether to generate a random environment')

    parser.add_argument("--seed", default=1e6, type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--save_models", action= "store",
        help='Whether or not models are saved')


    #--- Batch size is VERY important: 1024 is a winner ---
    parser.add_argument("--batch-size", default=32, type=int,
        help='Batch size for both actor and critic')
    #---

    parser.add_argument("--capacity", default=1e5, type=float,
        help='Size of replay buffer (bigger is better)')
    parser.add_argument("--compressed", default=False,
        action='store_true', dest='compressed',
        help='Use a compressed replay buffer for efficiency')

    parser.add_argument("--stack-size", default=1, type=int,
        help='How much history to use')
    parser.add_argument("--evaluation-episodes", default=3, type=int,
        help='How many times to evaluate actor')
    parser.add_argument("--profile", default=False, action="store_true",
        help="Profile the program for performance")
    parser.add_argument("--mode", default = 'image',
        help="[image|state|mixed]")
    parser.add_argument("--network", default = 'simple',
        help="Choose [simple|densenet]")
    parser.add_argument("--no-batchnorm", default = True, dest='batchnorm',
        action='store_false', help="Choose whether to use batchnorm")
    parser.add_argument("--dropout", default = False,
        action='store_true', help="Choose whether to use dropout")
    parser.add_argument("--deep", default = False,
        action='store_true', help="Use a deeper NN")
    parser.add_argument("--img-dim", default = 224, type=int,
        help="Size of img [224|64]")
    parser.add_argument("--img-depth", default = 3, type=int,
        help="Depth of image (1 for grey, 3 for RGB)")

    parser.add_argument("--name", default=None, type=str,
        help='Name to append to save directory')

    parser.add_argument("--discount", default=0.99, type=float,
        help='Discount factor (0.99 is good)')

    #--- Learning rates
    parser.add_argument("--lr", default=5e-5, type=float,
        help="Learning rate for critic optimizer")
    #---

    #--- Model save/load
    parser.add_argument("--load", default=None, type=str,
        help="Continue training from a subdir of ./logs/specific_model")
    parser.add_argument("--load-last", default=False, action='store_true',
        help="Continue training from last model")
    parser.add_argument("--load-best", default=False, action='store_true',
        help="If load-last is selected, continue from last best saved model")

    parser.add_argument("--n-samples", default=100, type=int,
            help="Number of samples for Batch DQN")

    parser.add_argument("--procs", default=1, type=int,
            help="Number of processes to spawn")

    parser.add_argument("--amp", default=False, action='store_true',
            help="Activate 16-bit support if available")

    parser.add_argument('--task', default=None, type=str,
            help="Task to carry out for env (reach|suture)")

    parser.add_argument('--hires', default=False, action='store_true',
            dest='hi_res_mode',
            help='Use hi-res images internally')
    parser.add_argument('--stereo', default=False, action='store_true',
            dest='stereo_mode',
            help='Use stereo images')
    parser.add_argument('--depthmap', default=False, action='store_true',
            dest='depthmap_mode',
            help='Use depth map from sim')

    parser.add_argument('--no-clean', default=True, action='store_false',
            dest='clean', help='Clean up previous envs')

    args = parser.parse_args()

    args.env = args.env.lower()
    # Image mode requires batchnorm
    args.batchnorm = True if args.mode in ['image', 'mixed'] else args.batchnorm
    args.img_dim = 64 if args.env == 'atari' else args.img_dim
    args.capacity = int(args.capacity)
    args.playback = True # playback is always true for test_cnn

    # Set default task
    if args.task is None:
        if args.env == 'sim':
            args.task = 'reach'
        elif args.env == 'needle':
            args.task = '15'

    assert (args.mode in ['image', 'mixed', 'state'])
    assert (args.network in ['simple', 'densenet'])

    if args.profile:
        import cProfile
        cProfile.run('run(args)', sort='cumtime')
    else:
        run(args)
