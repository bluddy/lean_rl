#!/usr/bin/env python
''' Parse the csv files for auxiliary losses and get their stats
    for normalization
'''

import os
import argparse
#import matplotlib.pyplot as plt
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('csv_file', help='csv file to read')
args = parser.parse_args()


data = [],[]

with open(args.csv_file, 'r') as f:
    csv_f = csv.reader(f, delimiter=',')
    target = 0
    for line in csv_f:
        data[target].extend(line)
        target = 1 - target

# Zipping allows us to examine all
data = list(zip(*data))

data2 = []
data3 = []

#import pdb
#pdb.set_trace()

for line1, line2 in data:
    def process(line):
        line = line.replace('\n','').replace('[','').replace(']','')
        line = line.strip().split(' ')
        line = filter(lambda x: x != '', line)
        line = np.array([float(x) for x in line], dtype=np.float32)
        return line
    data2.append(process(line1))
    data3.append(process(line2))

data, data2 = data2, data3

data3, data4 = [], []
for l1, l2 in zip(data, data2):
    assert len(l1) == 6 and len(l2) == 6
    data3.append(l1)
    data4.append(l2)

data, data2 = data3, data4

#data, data2 = data[-200:], data2[-200:] # comment if you want all

#import pdb
#pdb.set_trace()

data, data2 = np.array(data), np.array(data2)

print "data mean: {}\ndata stdev: {}\ndata2 mean: {}\ndata2 stdev: {}".format(
        np.mean(data, axis=0), np.std(data, axis=0), np.mean(data2, axis=0), np.std(data2, axis=0))

delta = np.abs(data - data2)

print "delta mean: {}\ndelta stdev: {}".format(
        np.mean(delta, axis=0), np.std(delta, axis=0))

print "done"




