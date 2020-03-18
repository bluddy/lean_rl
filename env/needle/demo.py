# -*- coding: utf-8 -*-
"""
Created on Thu Oct 08 11:31:19 2015

@author: Chris
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdb import set_trace as woah

'''
Stores data for a single performance of a task.
You can pull this data out as a Numpy array.
'''
class Demo:

    def __init__(self,env_width, env_height,filename=None):

        self.t = None
        self.s = None
        self.u = None
        ''' screen parameters for device demo was saved on '''
        self.device_width = None
        self.device_height = None

        self.env_height = env_height
        self.env_width = env_width

        if not filename is None:
            with open(filename, 'r') as file:
                (env, time) = self.parse_name(filename)
                self.load(file)
                self.env = env

    @staticmethod
    def parse_name(filename):
        toks = filename.split('/')[-1].split('.')[0].split('_')
        return (int(toks[1]),toks[2])

    def draw(self):
        plt.plot(self.s[:,0],self.s[:,1])

    '''
    Load demonstration from a file
    '''
    def load(self, handle):

        t = []
        s = []
        u = []

        data = handle.readline()
        while not data is None and len(data)>0:
            data = [float(x) for x in data.split(',')]

            t.append(data[0])
            s.append(data[1:4])
            u.append(data[4:])

            data = handle.readline()

        self.t = np.array(t)#.transpose()
        self.s = np.array(s)#.transpose()
        #self.s[1,:] = self.s[1,:]
        self.u = np.array(u)#.transpose()

    '''
    convert state, actions into environment coordinate frame
    from device coordinate frame
    '''
    def convert(self):
        width_ratio  = self.env_width / float(self.device_width)
        height_ratio = self.env_height / float(self.device_height)
        ''' update state information '''
        self.s[:, 0] = self.s[:, 0] * width_ratio
        self.s[:, 1] = self.s[:, 1] * height_ratio
        ''' update action information '''
        for t in range(len(self.u)):
            self.u[t,:] = self.convert_action(self.u[t,:])

    def convert_action(self, a):
        ''' TODO: vectorize this if we can '''
        r     = a[0]
        theta = a[1]
        width_ratio  = self.env_width / float(self.device_width)
        height_ratio = self.env_height / float(self.device_height)

        dx = r*math.cos(theta)
        dy = r*math.sin(theta)

        if(width_ratio == height_ratio):
            r_prime = width_ratio * r
            theta_prime = theta
        else:
            ''' TOOD implement if we have envs with diff aspect ratio'''
            pass

        # dx_prime = width_ratio * dx
        # dy_prime = height_ratio * dy
        #
        # r_prime = np.sqrt(dx_prime**2 + dy_prime**2)
        # if(r < 0):
        #     r_prime = -1*r_prime
        return np.array([r_prime, theta_prime])
