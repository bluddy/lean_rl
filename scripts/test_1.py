"""
    Set up reinforcement/imitation learning in the needle master environment

    * looking at the test.py code from Chris's work in tool_dir/scripts/


"""
import os
from .context import needlemaster as nm
import matplotlib.pyplot as plt
from pdb import set_trace as woah

""" set paths """
workspace_dir = '/mnt/c/Users/Molly/workspace/Surgical_Automation/'
tool_dir      = workspace_dir + 'src/needle_master_tools/'
exp_dir       = workspace_dir + 'experiments/needle_master_tools/'

""" select world """
env_dir     = exp_dir + 'environments/'
demo_dir    = exp_dir + 'demonstrations/'

env_number = 14
env_name   = 'environment_' + str(env_number) + '.txt'

""" load environment """
environment = nm.Environment(env_dir + env_name)

""" load in demonstrations """
demos = []

for file in os.listdir(demo_dir):
    (env,t) = nm.parse_demo_name(file)
    # only load demos from this environment
    if(env == env_number):
        demo = nm.Demo(env_height=environment.height,env_width=environment.width,filename=demo_dir + file)
        demos.append(demo)

environment.draw()
plt.gca().invert_xaxis()
plt.savefig('test_1_output.png')
plt.show()
# woah()

"""
===================================
Define rules of game:
    * have "needle"
    * moved with controls
    * cannot pierce some tissue


===================================
Define step:
* take action (dx, dy)
* render world
* increment time step t = t + 1
* update state


"""
def step(self, action):

    """ update variables """
    t = self.t[-1] + 1
    u = action
    s = [self.s[-1][0] + action[0], self.s[-1][1] + action[1],
        self.s[-1][2]]


    self.t.append(t)
    self.u.append(u)
    self.s.append(s)

    self.check_status()
    self.render_screen()

def check_status(self):
    """
        should we keep playing or did we fail/succeed?
    """

    # check if we've gone outside the boundaries of the game?
    # did we run out of time?
    # did we hit something illegal?

    if(we_pass_checks):
        self.terminated = False
    else:
        self.terminated = True
