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

# Total counts
total_times, total_loss, total_acc = [],[],[]

def run(args):

    last_learn_t, last_eval_t = 0, 0
    best_avg_reward = -1e5

    temp_q_avg, temp_q_max, temp_loss = [],[],[]
    timestep = 0
    warmup_t = args.learning_start

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
                save_mode_play_ratio=args.play_ratio,
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
                save_mode_play_ratio=args.play_ratio,
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

    action_steps = dummy_env.action_steps
    action_dim = dummy_env.action_dim

    img_depth = args.img_depth
    if args.stereo_mode or args.depthmap_mode:
        img_depth *= 2

    model = QImage(action_dim, bn=True, drop=args.dropout)

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
                warmup_t = args.learning_start
        model.load_state_dict(torch.load(pjoin(last_model_dir, 'model.pth')))
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
                warmup_t = args.learning_start
        print 'last_model_dir is {}, t={}'.format(last_model_dir, timestep)

    replay_buffer = CNNBuffer(args.mode, args.capacity, compressed=args.compressed)

    # Reset all envs and get first state
    for env in envs:
        env.reset()
    states = [env.get()[0] for env in envs] # block
    states_nd = dummy_env.combine_states(states)

    done = False

    elapsed_time = 0.

    proc_std = []
    terminate = False

    w_s, w_a, w_d, w_procs = [],[],[],[],[],[]

    dummy_action = np.zeros((action_dim,))

    # TOD:Place all envs in playback mode

    while timestep < args.max_timesteps and not terminate:

        acted = False
        start_t = time.time()
        new_states = []

        # Send non-blocking actions on ready envs
        for env in zip(envs):
            if env.is_ready():
                env.step(dummy_action)

        # Save our data so we can loop and insert it into the replay buffer
        for env, state in zip(envs, states):
            if env.is_ready() or env.poll():
                # Get the state and saved action from the env
                new_state, _, done, dict = env.get() # blocking
                action = dict["action"]
                best_action = dict["best_action"]
                new_states.append(new_state)

                if action is not None:
                    w_s.append(state)
                    w_a.append(best_action)

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
        if acted:
            #print "Time: ", elapsed_time # debug
            sys.stdout.write('.')
            sys.stdout.flush()
            elapsed_time = 0.

        # Evaluate episode
        if timestep - last_eval_t > args.eval_freq:

            last_eval_t = timestep

            print('\n---------------------------------------')
            print 'Evaluating CNN for ', logdir
            replay_buffer.display() # debug

            # Block and flush result if needed
            best_reward = evaluate_model(
                csv_wr, csv_f, log_f,
                tb_writer, logdir, total_times, total_rewards, total_loss,
                total_q_avg, total_q_max,
                temp_loss, temp_q_avg, temp_q_max,
                envs, args, model, timestep, test_path,
                last_learn_t, last_eval_t, best_avg_reward)

            # Restore envs
            for env in envs:
                if not env.is_ready(): # Flush old messages
                    env.get()
                env.reset()
            new_states = [env.get()[0] for env in envs]

            temp_loss = []

            def save_model(path):
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), pjoin(path, 'model.pth'))
                with open(pjoin(path, 'timestep.txt'), 'w') as f:
                    f.write(str(timestep))

            best_path = pjoin(model_path, 'best')
            if best_reward > best_avg_reward or not os.path.exists(best_path):
                best_avg_reward = best_reward
                print "Saving best avg reward: {}".format(best_avg_reward)
                save_model(best_path)
            save_model(model_path)

        ## Train
        if warmup_t <= 0 and \
            timestep - last_learn_t > args.learn_freq and \
            len(replay_buffer) > args.batch_size:

            last_learn_t = timestep

            model.train()

            # Train a few times
            for _ in range(1):
                beta = min(1.0, beta_start + timestep *
                    (1.0 - beta_start) / beta_frames)

                critic_loss, actor_loss, q_avg, q_max = policy.train(
                    replay_buffer, timestep, beta, args)

                temp_q_avg.append(q_avg)
                temp_q_max.append(q_max)
                temp_loss.append(critic_loss)

                save_mode_playing_cnt = 0
                save_mode_recording_cnt = 0
                for env in envs:
                    if env.get_save_mode() == 'play':
                        save_mode_playing_cnt += 1
                    if env.get_save_mode() == 'record':
                        save_mode_recording_cnt += 1

                if args.stop_after_playback and save_mode_playing_cnt == 0:
                    terminate = True

                s = '\nTraining T:{} TS:{:04d} CL:{:.5f} Exp_std:{:.2f} p:{} r:{}'.format(
                    str(datetime.timedelta(seconds=time.time() - program_start_t)),
                    timestep,
                    critic_loss,
                    0 if len(proc_std) == 0 else sum(proc_std)/len(proc_std),
                    save_mode_playing_cnt,
                    save_mode_recording_cnt
                    )
                s2 = ' AL:{:.5f}'.format(actor_loss) if actor_loss else ''
                print s + s2,
                proc_std = []

                log_f.write(s + s2 + '\n')

            model.eval()

        # print "Training done" # debug
        states = new_states
        states_nd = dummy_env.combine_states(states)

    print("Best Reward: ", best_reward)
    csv_f.close()
    log_f.close()

def evaluate_model(csv_wr, csv_f, log_f, tb_writer, logdir,
        temp_loss, temp_q_avg, temp_q_max,
        envs, args, model, timestep, test_path,
        last_learn_t, last_eval_t, best_avg_reward):
    ''' Evaluates model
        @param tb_writer: tensorboard writer
        @returns average_reward
    '''

    rewards = np.zeros((args.procs,), dtype=np.float32)
    actions = []
    num_done = 0
    for env in envs:
        env.set_save_mode('eval') # Stop recording
        if not env.is_ready(): # Flush old messages
            env.get()
        env.reset()
    states = [env.get()[0] for env in envs]
    while num_done < args.procs:
        acted = False
        # Send action
        for i, env in enumerate(envs):
            if not env.done:
                if env.is_ready():
                    action = policy.select_action(states[i]).squeeze(0)
                    actions.append(action)
                    env.step(action)
                    acted = True
                elif env.poll():
                    state, reward, done, _ = env.get()
                    if done:
                        num_done += 1
                    states[i] = state
                    rewards[i] += reward
                    env.render(save_path=test_path)
        if acted:
            sys.stdout.write('.')
            sys.stdout.flush()

    # Restore envs
    for env in envs:
        env.convert_to_video(save_path=test_path)
        env.restore_last_save_mode() # Resume recording

    avg_reward = rewards.mean()
    actions = np.array(actions, dtype=np.float32)
    avg_action = actions.mean()
    std_action = actions.std()
    min_action = actions.min()
    max_action = actions.max()

    total_times.append(timestep)
    total_times_nd = np.array(total_times)

    # Average over all the training we did since last timestep
    q_avg = np.mean(temp_q_avg) if len(temp_q_avg) > 0 else 0.
    q_max = np.max(temp_q_max) if len(temp_q_max) > 0 else 0.
    loss_avg = np.mean(temp_loss) if len(temp_loss) > 0 else 0.
    r_avg, r_var, r_low, r_up = utils.get_stats(total_rewards_nd)

    total_loss.append(loss_avg)

    fig = plt.figure()
    plt.plot(total_times_nd, total_q_avg, label='Average Q')
    plt.plot(total_times_nd, total_q_max, label='Max Q')
    plt.savefig(pjoin(logdir, 'q_avg_max.png'))
    tb_writer.add_figure('q_avg_max', fig, global_step=timestep)

    fig = plt.figure()
    plt.plot(total_times_nd, total_loss, label='Loss')
    plt.savefig(pjoin(logdir, 'loss_avg.png'))
    tb_writer.add_figure('loss_avg', fig, global_step=timestep)

    '''
    fig = plt.figure()
    plt.plot(total_times_nd, total_rewards_nd, label='Rewards')
    if len(total_rewards) > 100:
        # Plot average line
        plt.plot(total_times_nd[:len(avg_rewards)], avg_rewards)
    plt.savefig(pjoin(logdir, 'rewards.png'))
    tb_writer.add_figure('rewards', fig, global_step=timestep)
    '''

    fig = plt.figure()
    length = len(r_avg)
    plt.plot(total_times_nd[:length], r_avg, label='Rewards')
    #plt.fill_between(np.arange(len(r_avg)), r_low, r_up, alpha=0.4)
    plt.fill_between(total_times_nd[:length], r_low, r_up, alpha=0.4)
    plt.legend()
    plt.savefig(pjoin(logdir, 'rewards.png'))
    tb_writer.add_figure('rewards', fig, global_step=timestep)

    s = "\nEval Ep:{} R:{:.3f} Rav:{:.3f} BRav:{:.3f} a_avg:{:.2f} a_std:{:.2f} " \
        "min:{:.2f} max:{:.2f} " \
        "Q_avg:{:.2f} Q_max:{:.2f} loss:{:.3f}".format(
      env.episode, avg_reward, r_avg[-1], best_avg_reward, avg_action, std_action,
      min_action, max_action,
      q_avg, q_max, loss_avg)
    print s
    log_f.write(s + '\n')
    csv_wr.writerow([
        timestep, avg_reward, q_avg, q_max, loss_avg,
        best_avg_reward, last_learn_t, last_eval_t
    ])
    csv_f.flush()

    return r_avg[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--scale-rewards", default=False, action='store_true',
        help='Whether to scale rewards to be > 0')

    parser.add_argument("--seed", default=1e6, type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--eval-freq", default=3000, type=int, # 200
        help='How often (time steps) we evaluate')
    parser.add_argument("--learn-freq", default=100, type=int,
        help='Timesteps to explore before applying learning')
    parser.add_argument("--render-freq", default=100, type=int,
        help='How often (episodes) we save the images')
    parser.add_argument("--render-t-freq", default=5, type=int,
        help='How often (timesteps) we save the images in a saved episode')
    parser.add_argument("--max-timesteps", default=2e7, type=float,
        help='Max time steps to run environment for')
    parser.add_argument("--learning-start", default=None,
        help='Timesteps before learning')
    parser.add_argument("--save_models", action= "store",
        help='Whether or not models are saved')


    #--- Batch size is VERY important: 1024 is a winner ---
    parser.add_argument("--batch-size", default=32, type=int,
        help='Batch size for both actor and critic')
    #---

    parser.add_argument("--buffer", default = 'replay', # 'priority'
        help="Choose type of buffer, options are [replay, priority, disk, tier, tierpr]")
    parser.add_argument("--capacity", default=1e5, type=float,
        help='Size of replay buffer (bigger is better)')
    parser.add_argument("--compressed", default=False,
        action='store_true', dest='compressed',
        help='Use a compressed replay buffer for efficiency')
    parser.add_argument("--vacate-buffer", default=False, action='store_true',
        help='Vacate low priority items in the buffer first')
    parser.add_argument("--buffer-clip", default=100.,
        help='Clip Q in buffer')
    parser.add_argument('--sub-buffer', default='replay',
        help="Choose type of sub-buffer, options are [replay, priority]")

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
    parser.add_argument("--orig-q", default=False, action='store_true',
        help="Use original Q mechanism (from saved Q values)")

    parser.add_argument("--policy-freq", default=2, type=int,
        help='Frequency of TD3 delayed actor policy updates')
    parser.add_argument("--name", default=None, type=str,
        help='Name to append to save directory')

    parser.add_argument("--discount", default=0.99, type=float,
        help='Discount factor (0.99 is good)')

    #--- Tau: percent copied to target
    parser.add_argument("--tau", default=0.001, type=float,
        help='Target critic network update rate')
    parser.add_argument("--actor-tau", default=0.001, type=float,
        help='Target actor network update rate')
    #---

    #--- Learning rates
    parser.add_argument("--lr", default=5e-5, type=float,
        help="Learning rate for critic optimizer")
    parser.add_argument("--lr2", default=1e-3, type=float,
        help="Learning rate for second critic optimizer")
    parser.add_argument("--actor-lr", default=1e-5, type=float,
        help="Learning rate for actor optimizer")
    parser.add_argument("--clip-grad", default=None, type=float,
        help="Clip the gradient to slow down learning")
    #---

    #--- Model save/load
    parser.add_argument("--load-encoder", default='', type=str,
        help="File from which to load the encoder model")
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
    parser.add_argument('--reward', default='simple', type=str,
            help="Reward to use for the task (simple|v1)")


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

    if args.learning_start is None:
        args.learning_start = args.capacity
    else:
        args.learning_start = int(args.learning_start)

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
