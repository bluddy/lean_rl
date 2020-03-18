#!/usr/bin/env python

from .context import needlemaster as nm
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
from pdb import set_trace as woah

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='directory to read from')
args = parser.parse_args()

files = os.listdir(args.directory)

start_idx = 1
end_idx = 11
envs = [0]*(end_idx-start_idx)
ncols = 5

for i in range(start_idx,end_idx):
    file = "environment_%d.txt"%(i)

    # process as an environment
    env = nm.Environment(os.path.join(args.directory, file))
    envs[i-1] = env
    plt.subplot(2,ncols,i)
    env.draw()

for file in files:
    if file[:5] == 'trial':
        # process as a trial
        (env,t) = nm.ParseDemoName(file)

        # draw
        if env < end_idx and env >= start_idx:
            demo = nm.Demo(env_height=envs[env-1].height,env_width=envs[env-1].width,filename=os.path.join(args.directory, file))
            plt.subplot(2,ncols,env)
            demo.draw()
plt.savefig('test_output.png')
plt.show()
