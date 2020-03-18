import numpy as np
import torch
import random, math
import os, sys, argparse
from os.path import abspath
from os.path import join as pjoin
import plotly
from torch.utils.tensorboard import SummaryWriter

from models import ImageToPos

cur_dir= os.path.dirname(abspath(__file__))
sys.path.append(abspath(pjoin(cur_dir, '..')))
from needlemaster.environment import Environment

'''
Train the CNN to see the position of the needle, before we start
using RL
'''


def train(args):
    writer = SummaryWriter()

    mode = 'both'
    seed = 1e5

    env_data_name = os.path.splitext(
        os.path.basename(args.filename))[0]

    base_filename = '{}_{}'.format(args.env_name, env_data_name)

    # Set random seeds
    random.seed(seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(random.randint(1, 10000))
    else:
        args.device = torch.device('cpu')

    """ setting up environment """
    env = Environment(filename = args.filename, mode=mode,
        stack_size = args.stack_size, img_dim=args.img_dim)

    # Initialize policy
    action_dim = 1
    state_dim = len(env.gates) + 9

    out_size = 4
    model = ImageToPos(args.stack_size, out_size, img_dim=args.img_dim,
            bn=args.batchnorm).to(args.device)
    model.train() # for batchnorm

    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    max_iter = 100000
    render_freq = 500

    losses = []
    min_loss = 10000

    def save_model():
        torch.save(model.encoder.state_dict(), pjoin('models',
            'encoder_{}{}.pth'.format(args.img_dim,
                '_bn' if args.batchnorm else '')))

    for iter in xrange(max_iter):
        images, states = [], []
        iter_mult = iter * args.batch_size
        for j in xrange(args.batch_size):
            i, s = env.reset(random_needle=True)
            if (iter_mult + j) % render_freq == 0:
                env.render(save_image=True, save_path='./out_img/')
            #print s.shape, i.shape
            images.append(i)
            states.append(s[:, 0:4])
        images = np.array(images, dtype=np.float32)
        states = np.array(states, dtype=np.float32)
        #print images.shape, states.shape # debug
        x = torch.from_numpy(images).to(args.device)
        #print(states[0]) # debug
        y = torch.from_numpy(states).to(args.device).squeeze(1)

        y2 = model(x) # apply CNN
        loss = ((y2 - y) * (y2 - y))
        #print y2.shape, y.shape, loss.shape #debug
        loss2 = loss[0].clone().detach().cpu().sqrt().numpy()
        loss = loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        writer.add_scalar('Loss/train', loss, iter)

        if iter % args.save_freq == 0 and loss < min_loss:
            min_loss = loss
            save_model()

        print "Iter {} Loss = {:.5f}, {}".format(iter, loss, loss2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable-cuda', default=False, action='store_true',
        help='Disable CUDA')
    parser.add_argument("--env_name", default="NeedleMaster",
        help='OpenAI gym environment name')
    parser.add_argument("--seed", default=1e6, type=int,
        help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--pid_interval", default=5e3, type=int,
        help='How many time steps purely random policy is run for')
    parser.add_argument("--eval_freq", default=1e3, type=int,
        help='How often (time steps) we evaluate')
    parser.add_argument("--pid_freq", default=1e4, type=int,
        help='How often we get back to pure random action')
    parser.add_argument("--max_timesteps", default=5e6, type=float,
        help='Max time steps to run environment for')
    parser.add_argument("--learning_start", default=0, type=int,
        help='Timesteps before learning')
    parser.add_argument("--expl_noise", default=0.5, type=float,
        help='Starting std of Gaussian exploration noise')
    parser.add_argument("--epsilon_greedy", default=0.08, type=float,
        help='Starting percentage of choosing random noise')
    parser.add_argument("--batch-size", default=32, type=int,
        help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, type=float,
        help='Discount factor')
    parser.add_argument("--tau", default=0.005, type=float,
        help='Target network update rate')
    parser.add_argument("--policy_noise", default=0.2, type=float,
        help='Noise added to target policy during critic update')
    parser.add_argument("--noise_clip", default=0.5, type=float,
        help='Range to clip target policy noise')
    parser.add_argument("--policy_freq", default=2, type=int,
        help='Frequency of delayed policy updates')
    parser.add_argument("--max_size", default=5e4, type=int,
        help='Frequency of delayed policy updates')
    parser.add_argument("--stack-size", default=3, type=int,
        help='How much history to use')
    parser.add_argument("--evaluation_episodes", default=6, type=int)
    parser.add_argument("--profile", default=False, action="store_true",
        help="Profile the program for performance")
    parser.add_argument("--mode", default = 'state',
        help="Choose image or state, options are rgb_array and state")
    parser.add_argument("--img-dim", default = 224, type=int,
        help="Size of img (224 is max, 112/56 is optional)")
    parser.add_argument("--batchnorm", default = False,
        action='store_true', help="Choose whether to use batchnorm")
    parser.add_argument("--save-freq", default=500,
        help="How often to save the model")
    parser.add_argument("filename", help='File for environment')

    args = parser.parse_args()
    train(args)
