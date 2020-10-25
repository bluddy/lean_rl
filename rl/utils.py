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

def set_max_value_over_time(data):
    '''
    Track the max value over time and set it
    '''
    max = data[0]
    for i, x in enumerate(data):
        if x < max:
            data[i] = max
        else:
            max = x


class RateControl(object):
    ''' Stats for rate
        Assumes insertion of 1s and 0s
    '''
    def __init__(self, size=1000):
        self.data = np.zeros((size,), dtype=np.uint8)
        self.capacity = size
        self.size = 0
        self.pos = 0

    def add(self, x):
        self.data[self.pos] = x
        self.pos += 1
        if self.size < self.capacity:
            self.size += 1
        if self.pos >= self.capacity:
            self.pos = 0

    def rate(self):
        if self.size < self.capacity:
            r = np.mean(self.data[:self.size])
        else:
            r = np.mean(self.data)
        return r


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
