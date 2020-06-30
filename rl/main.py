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

cur_dir= os.path.dirname(abspath(__file__))

# Append one dir up to path
sys.path.append(abspath(pjoin(cur_dir, '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import utils
from buffers import *
import scipy.misc
from multiprocessing import Process, Pipe
from env_wrapper import EnvWrapper

def run(args):
    # Total counts
    total_times, total_rewards, total_q_avg, total_q_max, total_loss, total_measure = \
            [],[],[],[],[],[]

    last_learn_t, last_eval_t, last_stat_t = 0, 0, 0
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
        basename += '_' + args.mode[:2]
        if args.dropout:
            basename += '_drop'
        basename += '_' + args.task[:3]

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

    print "save_mode_path: ", save_mode_path
    # Save mode arguments
    save_mode = ''
    save_mode_post_play = ''
    if args.playback:
        save_mode = 'play'
    elif args.record:
        save_mode = 'record'

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
        # Disable nondeterministic ops (not sure if critical but better
        # safe than sorry)
        #torch.backends.cudnn.enabled = False
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
                save_mode_path=save_mode_path,
                save_mode=save_mode,
                save_mode_play_ratio=args.play_ratio,
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
    extra_state_dim = dummy_env.extra_state_dim

    img_depth = args.img_depth
    if args.depthmap_mode:
        img_depth += 1
    elif args.stereo_mode:
        img_depth *= 2

    if args.policy == 'td3':
        from policy.TD3 import TD3
        policy = TD3(state_dim, action_dim, args.stack_size,
            args.mode, lr=args.lr, img_depth=img_depth,
            bn=args.batchnorm, actor_lr=args.actor_lr, img_dim=args.img_dim)
    elif args.policy == 'ddpg':
        from policy.DDPG import DDPG
        policy = DDPG(state_dim, action_dim, args.stack_size,
            args.mode, lr=args.lr, img_depth=img_depth,
            bn=args.batchnorm, actor_lr=args.actor_lr, img_dim=args.img_dim)
    elif args.policy == 'dqn':
        from policy.DQN import DQN
        policy = DQN(state_dim, action_dim, action_steps, args.stack_size,
            args.mode, network=args.network, lr=args.lr,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp, dropout=args.dropout, aux=args.aux, aux_size=extra_state_dim,
            reduced_dim=args.reduced_dim, depthmap_mode=args.depthmap_mode, freeze=args.freeze)
    elif args.policy == 'ddqn':
        from policy.DQN import DDQN
        policy = DDQN(state_dim, action_dim, action_steps, args.stack_size,
            args.mode, network=args.network, lr=args.lr,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp, dropout=args.dropout, aux=args.aux, aux_size=extra_state_dim,
            reduced_dim=args.reduced_dim, depthmap_mode=args.depthmap_mode,
            freeze=args.freeze)
    elif args.policy == 'bdqn':
        from policy.DQN import BatchDQN
        policy = BatchDQN(state_dim=state_dim, action_dim=action_dim,
            action_steps=action_steps, stack_size=args.stack_size,
            mode=args.mode,
            n_samples=args.n_samples,
            network=args.network, lr=args.lr, bn=args.batchnorm,
            img_dim=args.img_dim, img_depth=img_depth,
            amp=args.amp)
    elif args.policy == 'dummy':
        from policy.dummy import Dummy
        policy = Dummy()
    else:
        raise ValueError(
            args.policy + ' is not recognized as a valid policy')

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
        policy.load(last_model_dir)
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
                total_rewards.append(r)
                total_q_avg.append(q_avg)
                total_q_max.append(q_max)
                total_loss.append(loss)
                csv_wr.writerow(line)
            csv_f.flush()
            if timestep is None:
                timestep = t + 1
                warmup_t = args.learning_start
        print 'last_model_dir is {}, t={}'.format(last_model_dir, timestep)

    ## load pre-trained policy
    #try:
    #    policy.load(result_path)
    #except:
    #    pass

    if args.buffer == 'replay':
        replay_buffer = ReplayBuffer(args.mode, args.capacity,
                compressed=args.compressed)
    elif args.buffer == 'priority':
        replay_buffer = NaivePrioritizedBuffer(args.mode, args.capacity,
                compressed=args.compressed, vacate=args.vacate_buffer)
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
    ou_noises = [utils.OUNoise(action_dim) for _ in range(args.procs)]

    policy.set_eval()

    elapsed_time = 0.

    proc_std = []
    terminate = False

    w_s1, w_s2, w_a, w_r, w_d, w_ba, w_es, w_procs = [],[],[],[],[],[],[],[]

    while timestep < args.max_timesteps and not terminate:

        # Interact with the environments

        # Check if we should add noise
        if args.ou_noise:
            noises = np.array([ou_noise.sample() for ou_noise in ou_noises])

        elif args.ep_greedy:
            # Epsilon-greedy
            percent_greedy = (1. - min(1., float(timestep) /
                greedy_decay_rate))
            epsilon_greedy = args.ep_greedy_pct * percent_greedy
            if random.random() < epsilon_greedy:
                noise_std = ((args.expl_noise - epsilon_final) *
                    math.exp(-1. * float(timestep) / std_decay_rate))
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
        #print "action2 proc std: ", np.std(actions2, axis=0)
        #print "actions2.shape: ", actions2.shape
        #print "actions: ", actions, " actions2: ", actions2 # debug

        # TODO: for dqn, we need to quantize the actions
        if args.policy == 'dqn':
            actions2 = policy.quantize_continuous(actions)
            #print "actions: ", actions, " actions2: ", actions2 # debug
            actions = actions2

        acted = False
        start_t = time.time()
        new_states = []

        # Send non-blocking actions on ready envs
        for env, action in zip(envs, actions):
            if env.is_ready():
                #print "XXX train action: ", action # debug
                env.step(action)

        #time.sleep(0.1)

        # Save our data so we can loop and insert it into the replay buffer
        for env, state, ou_noise in zip(envs, states, ou_noises):
            if env.is_ready() or env.poll():
                # Get the state and saved action from the env
                new_state, reward, done, d = env.get() # blocking
                action = d["action"]
                new_states.append(new_state)

                if action is not None:
                    w_s1.append(state)
                    w_s2.append(new_state)
                    w_r.append(reward)
                    w_d.append(done)
                    w_a.append(action)
                    ba = d["best_action"]
                    es = d["extra_state"]
                    w_ba.append(ba)
                    w_es.append(es)
                    w_procs.append(env.server_num)

                if done:
                    render_ep_path=None
                    if (env.episode + 1) % args.render_freq == 0:
                        render_ep_path=out_path

                    env.reset(render_ep_path=render_ep_path) # Send async action
                    ou_noise.reset() # Reset to mean

                acted = True
                if warmup_t <= 0:
                    timestep += 1
                else:
                    warmup_t -= 1
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
            for s1, s2, a, r, d, ba, es, p in zip(w_s1, w_s2, w_a, w_r, w_d, w_ba, w_es, w_procs):
                replay_buffer.add([s1, s2, a, r, d, ba, es], num=p)
            w_s1, w_s2, w_a, w_r, w_d, w_ba, w_es, w_procs = [],[],[],[],[],[],[],[]

        elapsed_time += time.time() - start_t
        if acted:
            #print "Time: ", elapsed_time # debug
            sys.stdout.write('.')
            sys.stdout.flush()
            elapsed_time = 0.

        # Evaluate episode
        if warmup_t <= 0 and timestep - last_eval_t > args.eval_freq:

            last_eval_t = timestep

            print('\n---------------------------------------')
            print 'Evaluating policy for ', logdir
            replay_buffer.display() # debug
            if args.ep_greedy:
                print("Greedy={}, std={}".format(epsilon_greedy, noise_std))

            # Block and flush result if needed
            best_reward = evaluate_policy(
                csv_wr, csv_f, log_f, tb_writer, logdir,
                total_times, total_rewards, total_loss, total_q_avg, total_q_max,
                temp_loss, temp_q_avg, temp_q_max,
                envs, args, policy, timestep, test_path,
                last_learn_t, last_eval_t, best_avg_reward)

            # Restore envs
            for env in envs:
                if not env.is_ready(): # Flush old messages
                    env.get()
                env.reset()
            new_states = [env.get()[0] for env in envs]

            temp_loss, temp_q_avg, temp_q_max = [], [], []

            def save_policy(path):
                if not os.path.exists(path):
                    os.makedirs(path)
                policy.save(path)
                with open(pjoin(path, 'timestep.txt'), 'w') as f:
                    f.write(str(timestep))

            best_path = pjoin(model_path, 'best')
            if best_reward > best_avg_reward or not os.path.exists(best_path):
                best_avg_reward = best_reward
                print "Saving best avg reward: {}".format(best_avg_reward)
                save_policy(best_path)
            save_policy(model_path)

            if args.aux is not None:
                test_cnn(policy, replay_buffer, total_times, total_measure, logdir, tb_writer,
                        args.eval_loops, log_f, timestep, csv_aux, args)

        ## Train
        if warmup_t <= 0 and \
            timestep - last_learn_t > args.learn_freq and \
            len(replay_buffer) > args.batch_size:

            last_learn_t = timestep

            policy.set_train()

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

            policy.set_eval()

        # print "Training done" # debug
        states = new_states
        states_nd = dummy_env.combine_states(states)

        ## Get stats
        if args.stat_freq != 0 and \
            timestep - last_stat_t > args.stat_freq and \
            len(replay_buffer) >= 50000:

            last_stat_t = timestep
            data = replay_buffer.sample(50000)
            data = data[0]
            avg = np.mean(data, axis=0)
            std = np.std(data, axis=0)

            s = '\nData mean:{}\n Data stdev:{}\n'.format(avg, std)
            print s
            log_f.write(s)


    print("Best Reward: ", best_reward)
    csv_f.close()
    log_f.close()

def test_cnn(policy, replay_buffer, total_times, total_measure, logdir, tb_writer, eval_loops, log_f,
        timestep, csv_aux, args):
    print 'Evaluating CNN for ', logdir
    test_loss, correct, total = [], 0, 0
    for _ in xrange(eval_loops):

        #import pdb
        #pdb.set_trace()

        x, pred_x = policy.test(replay_buffer, args)
        #csv_aux.writerow(x) # debug, takes up a lot of space
        #csv_aux.writerow(pred_x)
        #print action, predicted_action, '\n'
        if args.aux == 'action':
            correct += (x == pred_x).sum()
            total += len(action)
        elif args.aux == 'state':
            loss = (x - pred_x)
            loss = loss * loss
            loss = np.mean(loss)
            test_loss.append(loss)

    if args.aux == 'action':
        measure = correct / float(total)
        s = "Eval Accuracy: {:.3f}".format(measure)
        label = 'Accuracy'
    elif args.aux == 'state':
        measure = np.mean(test_loss)
        s = "Eval L2: {:.3f}".format(measure)
        label = 'L2 Dist'
    print s
    log_f.write(s + '\n')

    total_measure.append(measure)

    fig = plt.figure()
    plt.plot(total_measure, label=label)
    plt.savefig(pjoin(logdir, 'acc.png'))
    tb_writer.add_figure('acc', fig, global_step=timestep)

def evaluate_policy(
        csv_wr, csv_f, log_f, tb_writer, logdir,
        total_times, total_rewards, total_loss, total_q_avg, total_q_max,
        temp_loss, temp_q_avg, temp_q_max,
        envs, args, policy, timestep, test_path,
        last_learn_t, last_eval_t, best_avg_reward):
    ''' Runs deterministic policy for X episodes and
        @param tb_writer: tensorboard writer
        @returns average_reward
    '''

    #policy.actor.eval() # set for batchnorm
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

    total_rewards.append(avg_reward)
    total_rewards_nd = np.array(total_rewards)
    total_times.append(timestep)
    total_times_nd = np.array(total_times)

    # Average over all the training we did since last timestep
    q_avg = np.mean(temp_q_avg) if len(temp_q_avg) > 0 else 0.
    q_max = np.max(temp_q_max) if len(temp_q_max) > 0 else 0.
    loss_avg = np.mean(temp_loss) if len(temp_loss) > 0 else 0.
    r_avg, r_var, r_low, r_up = utils.get_stats(total_rewards_nd)

    total_q_avg.append(q_avg)
    total_q_max.append(q_max)
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
    parser.add_argument("--learning-start", default=0, type=int,
        help='Timesteps before learning')

    #--- Exploration Noise
    parser.add_argument("--no-ou-noise", default=True,
        action='store_false', dest='ou_noise',
        help='Use OU Noise process for noise instead of epsilon greedy')
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
    parser.add_argument("--capacity", default=1e5, type=float,
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
        help="Auxiliary loss: [state|action]")
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
    parser.add_argument("--load", default=None, type=str,
        help="Continue training from a subdir of ./logs/specific_model")
    parser.add_argument("--load-last", default=False, action='store_true',
        help="Continue training from last model")
    parser.add_argument("--load-best", default=False, action='store_true',
        help="If load-last is selected, continue from last best saved model")
    #---

    parser.add_argument("--policy", default="dqn", type=str,
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
    parser.add_argument('--playback', default=False, action='store_true',
            help='Play back the recorded data before running new simulation')
    parser.add_argument('--stop-after-playback', default=False, action='store_true',
            help='Stop after playback')
    parser.add_argument('--play-ratio', default=0, type=int,
            help='How many play episodes to real episodes to run')

    parser.add_argument('--record', default=False, action='store_true',
            dest='record',
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
        elif args.env == 'needle':
            args.task = '15'

    if args.env == 'sim' and args.task == 'reach':
        args.random_env = True

    assert not (args.depthmap_mode and args.stereo_mode)
    assert (args.mode in ['image', 'mixed', 'state'])
    assert (args.network in ['simple', 'densenet'])

    if args.profile:
        import cProfile
        cProfile.run('run(args)', sort='cumtime')
    else:
        run(args)
