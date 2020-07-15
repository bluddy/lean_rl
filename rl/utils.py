import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pdb

class OUNoise:
    '''Ornstein-Uhlenbeck process
       @param mu: mean
       @param theta: how much to reuse current state
       @param sigma: variance to add at each step
    '''
    def __init__(self, size, mu=0., theta=0.15, sigma=0.25):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu)'''
        self.state = self.mu[:]

    def sample(self):
        '''Update internal state and return as noise sample'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

def get_stats(data, div=50):
    '''
    @data: 1-d ndarray
    Returns: stats
    '''
    N = len(data) // div
    if N == 0:
        N = 1
    frac = np.ones((N,))/N
    avg = np.convolve(data, frac, mode='valid')
    var = np.convolve(data * data, frac, mode='valid')
    var = var - avg * avg
    std = np.sqrt(var)
    lower = avg - 3 * std
    upper = avg + 3 * std
    return avg, var, lower, upper


'''
class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin
'''

class ForkablePdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
