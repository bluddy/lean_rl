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
import json
from json import JSONEncoder, JSONDecoder

cur_dir= os.path.dirname(abspath(__file__))

# Append one dir up to path
sys.path.append(abspath(pjoin(cur_dir, '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

import rl.utils as utils
from .buffers import *
from .env_wrapper import EnvWrapper

import scipy.misc
from multiprocessing import Process, Pipe

class GlobalState(object):
    ''' Easy to serialize global state for runs '''
    def __init__(self, step=0, play_steps=0, best_reward=-1e5, last_train_step=0,
                last_eval_step=0, last_stat_step=0, total_reloads=0,
                consec_reloads=0, runtime=0, warmup_steps=0, reload_since_eval=False, **kwargs):
        self.step = step    # total steps
        self.play_steps = play_steps  # number of playback steps
        self.best_reward = best_reward
        self.last_train_step = last_train_step
        self.last_eval_step = last_eval_step
        self.last_stat_step = last_stat_step
        self.total_reloads = total_reloads
        self.consec_reloads = consec_reloads
        self.runtime = runtime # total runtime
        self.warmup_steps = warmup_steps
        self.reload_since_eval = reload_since_eval

class GlobalStateEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

def decode_globalstate(dct):
    return GlobalState(**dct)

def run(args):
    # Total counts
    total_times, total_rewards, total_q_avg, total_q_max, total_loss, total_measure, total_success1, \
    total_success2 = \
            [],[],[],[],[],[],[],[]

    # Reload constants
    min_reload_r = 2.
    max_consec_reloads = 4
    abs_r_delta_reload = 1.0
    rel_r_delta_reload = 0.4

    # Variables for rate control
    rate_control = utils.RateControl()
    # The more we sleep, the more the sim has time to catch up
    target_sleep_time = 0.
    delta_sleep_time = 0.005
    last_rate = 0.

    temp_q_avg, temp_q_max, temp_loss = [],[],[]
    env_time = 0.

    g = GlobalState()

    g.warmup_t = args.learning_start

    start_measure_time = time.time()

    if args.env == 'needle':
        from env.needle.env import Environment
        env_dir = pjoin(cur_dir, '..', 'env', 'needle', 'data')
        env_file = pjoin(env_dir, 'environment_' + args.task + '.txt')
        env_name = 'rand' if args.random_env else args.task

        env_suffix = '_a' + str(args.action_steps)
        if args.random_needle:
            env_suffix += 'r'

        basename = '{}_{}_{}_{}{}'.format(
            args.env, env_name, args.policy, args.mode[0:2], env_suffix)

        suffix = ''
        if args.random_env or args.random_needle:
            suffix += '_'
        if args.random_env:
            suffix += 'r'
        if args.random_needle:
            suffix += 'n'

        save_mode_path = os.path.join('saved_data', '{}_{}{}'.format(
            args.env, env_name, suffix))

    elif args.env == 'needle3d':
        from env.needle3d.env import Environment
        env_dir = pjoin(cur_dir, '..', 'env', 'needle', 'data')
        env_file = pjoin(env_dir, 'environment_' + args.task + '.txt')
        env_name = 'rand' if args.random_env else args.task


        basename = '{}_{}_{}_{}'.format(
            args.env, env_name, args.policy, args.mode[:2])
        if args.random_needle:
            basename += '_r'

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
        basename = args.env
        if args.random_env:
            basename += '_rt'
        basename += '_' + args.policy
        if args.aux is not None:
            basename += '_aux' + args.aux
        if args.stereo_mode:
            basename += '_s'
        elif args.depthmap_mode:
            basename += '_d'
        if args.amp:
            basename += '_amp'
        basename += '_' + args.mode[:2]
        if args.dropout:
            basename += '_drop'
        basename += '_' + args.task[:3]

        suffix = '_'
        if args.random_env:
            suffix += 'r'
        # We always save with _hsd, but don't always train/test with it
        suffix += 'hsd'

        save_mode_path = os.path.join('saved_data', '{}_{}{}'.format(
                args.env, args.task, suffix))

    elif args.env == 'atari':
        from env.atari.atari_env import Environment
        env_data_name = '_' + args.game

    else:
        raise ValueError("Unrecognized environment " + args.env)

    print("save_mode_path: ", save_mode_path)
    print("log_path: ", 'logs/' + basename)

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
    else:
        args.device = torch.device('cpu')

    ## environment setup
    log_f = open(pjoin(logdir, 'log.txt'), 'w')

    # CSV file for test resumption
    csv_f = open(pjoin(logdir, 'log.csv'), 'w')
    csv_wr = csv.writer(csv_f, delimiter=',')

    if args.aux is not None:
        csv_aux_f = open(pjoin(logdir, 'log_aux.csv'), 'w')
        csv_aux = csv.writer(csv_aux_f, delimiter=',')

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
                add_delay=args.add_delay,
                save_mode_path=save_mode_path,
                save_mode='',
                )

        elif args.env == 'needle3d':
            from env.needle3d.env import Environment
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
                add_delay=args.add_delay,
                save_mode_path=save_mode_path,
                save_mode='',
                full_init=not dummy_env,
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
                save_mode='',
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
    env_nums = list(range(args.procs))

    def set_save_mode_randomly():
        # Choose which envs to play, record, or n/a
        if args.playback == -1:
            nums = env_nums
        else:
            nums = random.sample(env_nums, args.playback)

        for i, env in enumerate(envs):
            if i in nums:
                env.set_save_mode('play')
            elif args.record:
                env.set_save_mode('record')
            else:
                env.set_save_mode('')

    set_save_mode_randomly()

    # Delays for resets, which are slow
    sleep_time = 0.2

    """ parameters for epsilon decay """
    greedy_decay_rate = 10000000
    std_decay_rate = 10000000
    epsilon_final = 0.001
    ep_decay = []

    """ beta Prioritized Experience Replay"""
    beta_start = 0.4
    beta_frames = 25000

    # Initialize policy
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
    extra_state_dim = 0
    if args.aux is not None:
        extra_state_dim = dummy_env.extra_state_dim

    img_depth = args.img_depth
    if args.depthmap_mode:
        img_depth += 1
    elif args.stereo_mode:
        img_depth *= 2

    if args.policy == 'td3':
        from policy.td3 import TD3
        policy = TD3(state_dim, action_dim, args.stack_size,
            args.mode, lr=args.lr, img_depth=img_depth,
            bn=args.batchnorm, actor_lr=args.actor_lr, img_dim=args.img_dim)
    elif args.policy == 'ddpg':
        from policy.DDPG import DDPG
        policy = DDPG(state_dim, action_dim, args.stack_size,
            args.mode, lr=args.lr, img_depth=img_depth,
            bn=args.batchnorm, actor_lr=args.actor_lr, img_dim=args.img_dim)
    elif args.policy == 'dqn':
        from rl.policy.dqn.dqn import DQN
        policy = DQN(state_dim, action_dim, action_steps, args.stack_size,
            args.mode, network=args.network, lr=args.lr,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp, dropout=args.dropout, aux=args.aux, aux_size=extra_state_dim,
            reduced_dim=args.reduced_dim, depthmap_mode=args.depthmap_mode, freeze=args.freeze, opt=args.opt)
    elif args.policy == 'ddqn':
        from rl.policy.dqn.dqn import DDQN
        policy = DDQN(state_dim=state_dim, action_dim=action_dim, action_steps=action_steps,
                stack_size=args.stack_size, mode=args.mode, network=args.network, lr=args.lr,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp, dropout=args.dropout, aux=args.aux, aux_size=extra_state_dim,
            reduced_dim=args.reduced_dim, depthmap_mode=args.depthmap_mode,
            freeze=args.freeze, opt=args.opt)
    elif args.policy == 'dummy':
        from policy.dummy import Dummy
        policy = Dummy()
    else:
        raise ValueError(
            args.policy + ' is not recognized as a valid policy')

    # Load from files if requested
    if args.load_last and last_dir is not None:
        model_dir = pjoin(logbase, last_dir, 'model')
        if args.load_best:
            model_dir = pjoin(model_dir, 'best')
        data_file = pjoin(model_dir, 'data.json')
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                g = json.load(f, object_hook=decode_globalstate)
        policy.load(model_dir)
        # Same csv file for best and last
        csv_file = pjoin(logbase, last_dir, 'log.csv')
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            t = 0
            # load data and rewrite csv
            for line in reader:
                t = int(line[0])
                if t > g.step:
                    break
                r, q_avg, q_max, loss, _ = map(lambda x: float(x), line[1:6])
                if len(line) > 8:
                    succ1_pct = float(line[8])
                    succ2_pct = float(line[9])
                    # ignore play_steps -- we're not graphing that automatically
                total_times.append(t)
                total_rewards.append(r)
                total_q_avg.append(q_avg)
                total_q_max.append(q_max)
                total_loss.append(loss)
                total_success1.append(succ1_pct)
                total_success2.append(succ2_pct)
                csv_wr.writerow(line)
            csv_f.flush()
        print('model_dir is {}, t={}'.format(model_dir, g.step))

    if args.buffer == 'replay':
        replay_buffer = ReplayBuffer(args.mode, args.capacity,
                compressed=args.compressed)
    elif args.buffer == 'priority':
        replay_buffer = NaivePrioritizedBuffer(args.mode, args.capacity,
                compressed=args.compressed)
    elif args.buffer == 'multi':
        replay_buffer = MultiBuffer(args.mode, args.capacity,
                compressed=args.compressed, sub_buffer='priority')
    elif args.buffer == 'disk':
        replay_buffer = DiskReplayBuffer(args.mode, args.capacity, logdir)
    elif args.buffer == 'tier':
        replay_buffer = TieredBuffer(args.mode, args.capacity,
                compressed=args.compressed, procs=args.procs,
                clip=args.buffer_clip, sub_buffer='replay')
    elif args.buffer == 'tierpr':
        replay_buffer = TieredBuffer(args.mode, args.capacity,
                compressed=args.compressed, procs=args.procs,
                clip=args.buffer_clip, sub_buffer='priority')
    else:
        raise ValueError(args.buffer + ' is not a buffer name')

    # Reset all envs and get first state
    for env in envs:
        env.reset()
    states = [env.get()[0] for env in envs] # block
    states_nd = dummy_env.combine_states(states)

    zero_noises = np.zeros((args.procs, action_dim))
    ou_noises = [utils.OUNoise(action_dim, theta=args.ou_theta, sigma=args.ou_sigma) \
            for _ in range(args.procs)]

    policy.set_eval()

    proc_std = []
    terminate = False

    # Termporary storage for data
    w_s1, w_s2, w_a, w_r, w_d, w_es, w_procs = [],[],[],[],[],[],[]

    while g.step < args.max_timesteps and not terminate:

        # Interact with the environments

        # Check if we should add noise
        if args.ou_noise:
            noises = np.array([ou_noise.sample() for ou_noise in ou_noises])

        elif args.ep_greedy:
            # Epsilon-greedy
            percent_greedy = (1. - min(1., float(g.step) /
                greedy_decay_rate))
            epsilon_greedy = args.ep_greedy_pct * percent_greedy
            if random.random() < epsilon_greedy:
                noise_std = ((args.expl_noise - epsilon_final) *
                    math.exp(-1. * float(g.step) / std_decay_rate))
                ep_decay.append(noise_std)
                # log_f.write('epsilon decay:{}\n'.format(noise_std)) # debug
                noise = np.random.normal(0, noise_std, size=action_dim)
            else:
                noises = zero_noises
        else:
            noises = zero_noises

        """ action selected based on pure policy """
        actions2 = policy.select_action(states_nd)

        # Track stdev of chosen actions between procs
        proc_std.append(np.mean(np.std(actions2, axis=0)))

        actions = np.clip(actions2 + noises, -1., 1.)

        # Debug stuff
        #print("action2 proc std: ", np.std(actions2, axis=0))
        #print("actions2.shape: ", actions2.shape)
        #print("actions: ", actions, " actions2: ", actions2) # debug

        # for dqn, we need to quantize the actions
        if args.policy == 'dqn':
            actions2 = policy.quantize_continuous(actions)
            #print("actions: ", actions, " actions2: ", actions2) # debug
            actions = actions2

        acted = False
        env_measure_time = time.time()
        new_states = []

        # Send non-blocking actions on ready envs
        for env, action in zip(envs, actions):
            if env.is_ready():
                #print("XXX train action: ", action) # debug
                env.step(action)

        if target_sleep_time > 0:
            time.sleep(target_sleep_time)

        # Save our data so we can loop and insert it into the replay buffer
        for env, state, ou_noise in zip(envs, states, ou_noises):
            if env.is_ready() or env.poll():
                # Get the state and saved action from the env
                new_state, reward, done, d = env.get() # blocking
                action = d["action"]
                new_states.append(new_state)

                if action is not None: # Not reset
                    w_s1.append(state)
                    w_s2.append(new_state)
                    w_r.append(reward)
                    w_d.append(done)
                    w_a.append(action)
                    w_es.append(d["extra_state"] if "extra_state" in d else None)
                    w_procs.append(env.server_num)

                if done:
                    render_ep_path=None
                    if (env.episode + 1) % args.render_freq == 0:
                        render_ep_path=out_path

                    env.reset(render_ep_path=render_ep_path) # Send async action
                    ou_noise.reset() # Reset to mean

                acted = True
                if g.warmup_steps <= 0:
                    g.step += 1
                    if env.get_save_mode() == 'play':
                        g.play_steps += 1
                        rate_control.add(1.)
                    else:
                        rate_control.add(0.)
                else:
                    g.warmup_steps -= 1
            else:
                # Nothing to do with the env yet
                new_states.append(state)

        if len(w_s1) > 10:
            # Do compression in parallel
            if args.compressed:
                w_s1 = Parallel(n_jobs=-1)(delayed(state_compress)
                        (args.mode, s) for s in w_s1)
                w_s2 = Parallel(n_jobs=-1)(delayed(state_compress)
                        (args.mode, s) for s in w_s2)
                '''
                w_s1 = [state_compress(args.mode, s) for s in w_s1]
                w_s2 = [state_compress(args.mode, s) for s in w_s2]
                '''

            # Feed into the replay buffer
            for s1, s2, a, r, d, es, p in zip(w_s1, w_s2, w_a, w_r, w_d, w_es, w_procs):
                data = [s1, s2, a, r, d]
                if es is not None:
                    data.append(es)
                replay_buffer.add(data, num=p)
            w_s1, w_s2, w_a, w_r, w_d, w_es, w_procs = [],[],[],[],[],[],[]

        env_time += time.time() - env_measure_time
        if acted:
            #print("Env time: ", env_elapsed_time) # debug
            sys.stdout.write('.')
            sys.stdout.flush()
            env_time = 0.

        # Evaluate episode
        if g.warmup_steps <= 0 and g.step - g.last_eval_step > args.eval_freq:

            g.last_eval_step = g.step

            print('\n---------------------------------------')
            print('Evaluating policy for ', logdir)
            replay_buffer.display() # debug
            if args.ep_greedy:
                print("Greedy={}, std={}".format(epsilon_greedy, noise_std))

            # Block and flush result if needed
            new_reward = evaluate_policy(
                csv_wr, csv_f, log_f, tb_writer, logdir,
                total_times, total_rewards, total_loss, total_q_avg, total_q_max, total_success1,
                total_success2,
                temp_loss, temp_q_avg, temp_q_max,
                envs, args, policy, g, test_path)
            g.reload_since_eval = False

            # Set save mode randomly for the environments
            set_save_mode_randomly()

            # Restore envs from evaluation
            for env in envs:
                if not env.is_ready(): # Flush old messages
                    env.get()
                env.reset()
            new_states = [env.get()[0] for env in envs]

            temp_loss, temp_q_avg, temp_q_max = [], [], []

            # Update runtime
            cur_time = time.time()
            g.runtime += cur_time - start_measure_time
            start_measure_time = cur_time

            def save_policy(path):
                if not os.path.exists(path):
                    os.makedirs(path)
                policy.save(path)
                with open(pjoin(path, 'data.json'), 'w') as f:
                    json.dump(g, f, cls=GlobalStateEncoder)

            best_path = pjoin(model_path, 'best')
            if new_reward > g.best_reward or not os.path.exists(best_path):
                g.best_reward = new_reward
                print("Saving best reward model: R={}".format(g.best_reward))
                save_policy(best_path)

            # Check if we regressed by more than X% from max.
            # If so, reload the model, but don't reset timesteps
            # After x consecutive reloads, give up
            if args.autoreload and \
               g.consec_reloads < max_consec_reloads and \
               g.best_reward > min_reload_r and \
               new_reward < g.best_reward and \
               abs(g.best_reward - new_reward) > abs_r_delta_reload and \
               abs(g.best_reward - new_reward) / abs(g.best_reward) > rel_r_delta_reload:
                   g.total_reloads += 1
                   g.consec_reloads +=1
                   print("!!Reloading best model {} times, conseq {}: Best reward:{:.3f}, new reward:{:.3f}, high drop.".format(g.total_reloads, g.consec_reloads, g.best_reward, new_reward))
                   # Only load policy, not any global vars
                   policy.load(best_path)
                   g.reload_since_eval = True
            else:
                # no reloads
                g.consec_reloads = 0
                save_policy(model_path) # Always save

            # Training aux if needed
            if args.aux is not None:
                csv_aux_arg = csv_aux if args.aux_collect else None
                test_cnn(policy, replay_buffer, total_times, total_measure, logdir, tb_writer,
                        args.eval_loops, log_f, g, csv_aux_arg, args)

        ## Train model
        if g.warmup_steps <= 0 and \
            g.step - g.last_train_step > args.train_freq and \
            len(replay_buffer) > args.batch_size:

            g.last_train_step = g.step

            # Check rate
            if args.play_rate != 0.:
                rate = rate_control.rate() * 100.
                if rate > args.play_rate + 0.05 and rate >= last_rate:
                    target_sleep_time += delta_sleep_time
                elif rate < args.play_rate - 0.05 and rate <= last_rate:
                    target_sleep_time -= delta_sleep_time
                last_rate = rate

                if target_sleep_time < 0:
                    target_sleep_time = 0

            policy.set_train()

            # Train a few times
            for _ in range(1):
                beta = min(1.0, beta_start + g.step *
                    (1.0 - beta_start) / beta_frames)

                critic_loss, actor_loss, q_avg, q_max = policy.train(
                    replay_buffer, g.step, batch_size=args.batch_size,
                    discount=args.discount, tau=args.tau, beta=beta)

                temp_q_avg.append(q_avg)
                temp_q_max.append(q_max)
                temp_loss.append(critic_loss)

                # Collect stats
                play_cnt, rec_cnt, sim_cnt = 0, 0, 0
                for env in envs:
                    mode = env.get_save_mode()
                    if mode == 'play':
                        play_cnt += 1
                    elif mode == 'record':
                        rec_cnt += 1
                    else:
                        sim_cnt += 1

                s = '\nTraining T:{} TS:{:04d} PTS%:{:.1f} CL:{:.5f} Exp_std:{:.2f} p{}r{}s{}'.format(
                    str(datetime.timedelta(seconds=g.runtime + time.time() - start_measure_time)),
                    g.step,
                    rate_control.rate() * 100.,
                    critic_loss,
                    0 if len(proc_std) == 0 else sum(proc_std)/len(proc_std),
                    play_cnt, rec_cnt, sim_cnt
                    )
                s2 = ' AL:{:.5f}'.format(actor_loss) if actor_loss else ''
                print(s + s2, end='')
                proc_std = []

                log_f.write(s + s2 + '\n')

            policy.set_eval()

        # print("Training done") # debug
        states = new_states
        states_nd = dummy_env.combine_states(states)

        ## Get stats
        if args.stat_freq != 0 and \
            g.step - g.last_stat_step > args.stat_freq and \
            len(replay_buffer) >= 50000:

            g.last_stat_step = g.step
            data = replay_buffer.sample(50000)
            data = data[0]
            avg = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            s = '\nData mean:{}\n Data stdev:{}\n'.format(avg, std)
            print(s)
            log_f.write(s)


    print("Best Reward: ", g.best_reward)
    csv_f.close()
    log_f.close()

def test_cnn(policy, replay_buffer, total_times, total_measure, logdir, tb_writer, eval_loops, log_f,
        g, csv_aux, args):
    print('Evaluating CNN for ', logdir)
    test_loss, correct, total = [], 0, 0
    for _ in range(eval_loops):

        #import pdb
        #pdb.set_trace()

        x, pred_x = policy.test(replay_buffer, args.batch_size)
        if csv_aux is not None:
            csv_aux.writerow(x) # debug, takes up a lot of space
            csv_aux.writerow(pred_x)
        #print(action, predicted_action, '\n')
        if args.aux == 'state':
            loss = (x - pred_x)
            loss = loss * loss
            loss = np.mean(loss)
            test_loss.append(loss)

    if args.aux == 'state':
        measure = np.mean(test_loss)
        s = "Eval L2: {:.3f}".format(measure)
        label = 'L2 Dist'
    print(s)
    log_f.write(s + '\n')

    total_measure.append(measure)

    fig = plt.figure()
    plt.plot(total_measure, label=label)
    plt.savefig(pjoin(logdir, 'acc.png'))
    tb_writer.add_figure('acc', fig, global_step=g.step)
    plt.close()

def evaluate_policy(
        csv_wr, csv_f, log_f, tb_writer, logdir,
        total_times, total_rewards, total_loss, total_q_avg, total_q_max, total_success1,
        total_success2,
        temp_loss, temp_q_avg, temp_q_max,
        envs, args, policy, g, test_path):
    ''' Runs deterministic policy for X episodes and
        @param tb_writer: tensorboard writer
        @returns average_reward
    '''

    #policy.actor.eval() # set for batchnorm
    rewards = np.zeros((args.procs,), dtype=np.float32)
    actions = []
    success_1 = []
    success_2 = []
    for env in envs:
        env.set_save_mode('eval') # Stop recording
        if not env.is_ready(): # Flush old messages
            env.get()
        env.reset()
    states = [env.get()[0] for env in envs]
    running = True
    while running:
        running = False
        acted = False
        # Send action
        for i, env in enumerate(envs):
            if not env.done:
                running = True
                if env.is_ready():
                    action = policy.select_action(states[i]).squeeze(0)
                    actions.append(action)
                    env.step(action)
                    acted = True
                elif env.poll():
                    state, reward, done, _ = env.get()
                    states[i] = state
                    rewards[i] += reward
                    env.render(save_path=test_path)
        if acted:
            sys.stdout.write('.')
            sys.stdout.flush()

    succ_temp = [0 for _ in range(3)] # All ending states
    for i, env in enumerate(envs):
        if env.success < 0:
            env.success = 0
        elif env.success > 2:
            env.success = 2
        succ_temp[env.success] += 1

    #Get average % for success states 1 and 2
    succ_sum = sum(succ_temp)
    succ1_pct = succ_temp[1] / float(succ_sum)
    succ2_pct = succ_temp[2] / float(succ_sum)

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

    total_rewards.append(avg_reward)
    total_rewards_nd = np.array(total_rewards)
    r_avg, r_var, r_low, r_high = utils.get_stats(total_rewards_nd)

    total_success1.append(succ1_pct)
    total_success1_nd = np.array(total_success1)
    succ1_avg, succ1_var, _, _ = utils.get_stats(total_success1_nd)
    total_success2.append(succ2_pct)
    total_success2_nd = np.array(total_success2)
    use_succ2 = np.any(total_success2_nd != 0.)
    if use_succ2:
        succ2_avg, succ2_var, _, _ = utils.get_stats(total_success2_nd)

    total_times.append(g.step)
    total_times_nd = np.array(total_times)

    #-- Average over all the training we did since last timestep
    q_avg = np.mean(temp_q_avg) if len(temp_q_avg) > 0 else 0.
    q_max = np.max(temp_q_max) if len(temp_q_max) > 0 else 0.
    loss_avg = np.mean(temp_loss) if len(temp_loss) > 0 else 0.

    total_q_avg.append(q_avg)
    total_q_max.append(q_max)
    total_loss.append(loss_avg)

    ## Plot Q
    fig = plt.figure()
    plt.plot(total_times_nd, total_q_avg, label='Average Q')
    plt.plot(total_times_nd, total_q_max, label='Max Q')
    plt.savefig(pjoin(logdir, 'q_avg_max.png'))
    tb_writer.add_figure('q_avg_max', fig, global_step=g.step)
    plt.close()
    #--

    ## Plot loss
    fig = plt.figure()
    plt.plot(total_times_nd, total_loss, label='Loss')
    plt.savefig(pjoin(logdir, 'loss_avg.png'))
    tb_writer.add_figure('loss_avg', fig, global_step=g.step)
    plt.close()

    ## Plot rewards
    fig = plt.figure()
    length = len(r_avg)
    plt.plot(total_times_nd[:length], r_avg, label='Rewards')
    plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.savefig(pjoin(logdir, 'rewards.png'))
    tb_writer.add_figure('rewards', fig, global_step=g.step)
    plt.close()

    ## Plot success
    fig = plt.figure()
    length = len(succ1_avg)
    if use_succ2:
        plt.plot(total_times_nd[:length], succ1_avg, label='State 1')
        plt.plot(total_times_nd[:length], succ2_avg, label='State 2')
    else:
        plt.plot(total_times_nd[:length], succ1_avg, label='Success')
    plt.savefig(pjoin(logdir, 'success.png'))
    plt.close()

    s = "\nEval Ep:{} R:{:.3f} Rav:{:.3f} BRav:{:.3f} a_avg:{:.2f} a_std:{:.2f} " \
        "min:{:.2f} max:{:.2f} " \
        "Q_avg:{:.2f} Q_max:{:.2f} loss:{:.3f}".format(
      env.episode, avg_reward, r_avg[-1], g.best_reward, avg_action, std_action,
      min_action, max_action, q_avg, q_max, loss_avg)
    print(s)
    log_f.write(s + '\n')
    # TODO: remove last_learn_t, last_eval_t, g.best_reward
    csv_wr.writerow([
        g.step, avg_reward, q_avg, q_max, loss_avg,
        g.best_reward, g.last_train_step, g.last_eval_step, succ1_pct, succ2_pct, g.play_steps,
        1 if g.reload_since_eval else 0
    ])
    csv_f.flush()

    mean_last_rewards = np.mean(total_rewards_nd[-5:])
    return float(mean_last_rewards)

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
    parser.add_argument("--eval-freq", default=10000, type=int, # 200
        help='How often (time steps) we evaluate')
    parser.add_argument("--train-freq", default=100, type=int,
        help='Timesteps to explore before applying learning')
    parser.add_argument("--render-freq", default=100, type=int,
        help='How often (episodes) we save the images')
    parser.add_argument("--render-t-freq", default=5, type=int,
        help='How often (timesteps) we save the images in a saved episode')
    parser.add_argument("--max-timesteps", default=2e7, type=float,
        help='Max time steps to run environment for')
    parser.add_argument("--learning-start", default=0, type=int,
        help='Timesteps before learning')

    #--- Exploration Noise
    parser.add_argument("--no-ou-noise", default=True,
        action='store_false', dest='ou_noise',
        help='Use OU Noise process for noise instead of epsilon greedy')
    parser.add_argument("--ou-sigma", default=0.5, type=float,
        help='OU sigma level: how much to add per step') # was 0.25
    parser.add_argument("--ou-theta", default=0.15, type=float,
        help='OU theta: how much to reuse current levels')



    parser.add_argument("--ep-greedy", default=False,
        action='store_true',
        help='Use epsilon greedy')
    parser.add_argument("--expl-noise", default=1., type=float,
        help='Starting std of Gaussian exploration noise')
    parser.add_argument("--ep-greedy-pct", default=0.3, type=float,
        help='Starting percentage of choosing random noise')
    #---

    #--- Batch size is VERY important: 1024 is a winner ---
    parser.add_argument("--batch-size", default=32, type=int,
        help='Batch size for both actor and critic')
    #---
    parser.add_argument("--policy-noise", default=0.04, type=float, # was 0.2
        help='TD3 Smoothing noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.1, type=float,
        help='TD3 Range to clip target policy noise') # was 0.5

    parser.add_argument("--buffer", default = 'replay', # 'priority'
        help="Choose type of buffer, options are [replay, priority, disk, tier, tierpr]")
    parser.add_argument("--capacity", default=5e4, type=float,
        help='Size of replay buffer (bigger is better)')
    parser.add_argument("--compressed", default=False,
        action='store_true', dest='compressed',
        help='Use a compressed replay buffer for efficiency')
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
    parser.add_argument("--dropout", default = False,
        action='store_true', help="Choose whether to use dropout")

    parser.add_argument("--deep", default = False,
        action='store_true', help="Use a deeper NN")
    parser.add_argument("--img-dim", default = 224, type=int,
        help="Size of img [224|128|64]")
    parser.add_argument("--img-depth", default = 3, type=int,
        help="Depth of image (1 for grey, 3 for RGB)")

    parser.add_argument("--aux", default=None, type=str,
        help="Auxiliary loss: [state]")
    parser.add_argument("--no-aux", default=False, action='store_true',
        help="No auxiliary loss")
    parser.add_argument("--aux-collect", default=False, action='store_true',
        help="Collect data for auxiliary loss")
    parser.add_argument("--reduced-dim", default = 100, type=int,
            help="Bottleneck for neural network (default:100)")

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

    #--- Optimizer
    parser.add_argument("--opt", default='adam',
        help="Optimizer to use [sgd|adam]")
    parser.add_argument("--lr", default=5e-5, type=float,
        help="Learning rate for critic optimizer (sgd:1e-3, adam:5e-5)")
    parser.add_argument("--lr2", default=1e-3, type=float,
        help="Learning rate for second critic optimizer")
    parser.add_argument("--actor-lr", default=1e-5, type=float,
        help="Learning rate for actor optimizer")
    parser.add_argument("--clip-grad", default=None, type=float,
        help="Clip the gradient to slow down learning")
    #---

    #--- Model save/load
    parser.add_argument("--load", default=None, type=str,
        help="Continue training from a subdir of ./logs/specific_model")
    parser.add_argument("--load-last", default=False, action='store_true',
        help="Continue training from last model")
    parser.add_argument("--load-best", default=False, action='store_true',
        help="If load-last is selected, continue from last best saved model")
    #---

    parser.add_argument("--policy", default="ddqn", type=str,
            help="Policy type. dummy|ddpg|td3|dqn|ddqn|bdqn")
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


    # -- save mode
    parser.add_argument('--playback', default=0, type=int,
            help='Play back the recorded data, -1 is all sims')
    parser.add_argument('--record', default=False, action='store_true',
            help='Record data from the experiment for future playback')
    # --

    parser.add_argument('--hires', default=False, action='store_true',
            dest='hi_res_mode',
            help='Use hi-res images internally')
    parser.add_argument('--stereo', default=False, action='store_true',
            dest='stereo_mode',
            help='Use stereo images')
    parser.add_argument('--depthmap', default=False, action='store_true',
            dest='depthmap_mode',
            help='Use depth map from sim')
    parser.add_argument('--no-freeze', default=True, action='store_false',
            dest='freeze',
            help='Freeze the models halfway')

    parser.add_argument('--no-clean', default=True, action='store_false',
            dest='clean', help='Clean up previous envs')
    #-- Test-cnn for aux
    parser.add_argument('--eval-loops', default=100, type=int,
            help='How many times to test over data in replay buffer')

    #-- find stats of data
    parser.add_argument('--stat-freq', default=0, type=int,
            help='How often to evaluate statistics from replay buffer')

    parser.add_argument('--autoreload', default=False, action='store_true',
            dest='autoreload',
            help='Reload when quality drops')

    parser.add_argument('--play-rate', default=0., type=float,
            help='Dynamic % playing to aim for')

    parser.add_argument('--add-delay', default=0., type=float,
            help='Add delay to fast environments')

    args = parser.parse_args()

    args.policy = args.policy.lower()
    args.env = args.env.lower()
    # Image mode requires batchnorm
    args.batchnorm = True
    args.img_dim = 64 if args.env == 'atari' else args.img_dim
    args.capacity = int(args.capacity)


    # Set default task
    if args.task is None:
        if args.env == 'sim':
            args.task = 'reach'
        elif args.env in ['needle', 'needle3d']:
            args.task = '15'

    if args.env == 'sim' and args.task == 'reach':
        args.random_env = True

    if args.env == 'sim' and args.mode != 'state' and not args.no_aux:
        args.aux = 'state'

    assert not (args.depthmap_mode and args.stereo_mode)
    assert (args.mode in ['image', 'mixed', 'state'])
    assert (args.network in ['simple', 'densenet'])

    if args.profile:
        import cProfile
        cProfile.run('run(args)', sort='cumtime')
    else:
        run(args)
