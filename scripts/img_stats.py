#!/usr/bin/env python
''' Parse the img files in saved_date to get stats (mean, stdev)
    for normalization.
    Only for depth images
'''

import os
import argparse
#import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dir', help='dir for files')
args = parser.parse_args()


g_mean_d = 0.
g_stdev_d = 0.
g_mean_i = 0.
g_stdev_i = 0.
num_samples = 0

files = os.listdir(args.dir)

dim = 224 * 2
width = dim * 3
height = dim
e24 = pow(2,24)

for file in files:
    # Open video file
    command = [ 'ffmpeg',
        '-loglevel', '8',
        '-i', file,
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo', '-'
        ]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    images = []
    while True:
        raw_image = pipe.stdout.read(width * height * 3)
        if len(raw_image) == 0:
            print "error: no raw image"
            sys.exit(1)
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((img_height, img_width, 3))
        images.append(image)

    images = np.concatenate(images, axis=0)
    # Split up into left, depth
    left = image[:, :,:dim,:]
    depth = image[:, :,(dim*2):,:].astype(np.int32)

    # Convert img to float32
    img = left.astype(np.float32)
    img /= 255.

    std = np.std(img)
    mean = np.mean(img)

    # Convert depth to float32
    depth[:,:,:,0] |= depth[:,:,:,1] << 8
    depth[:,:,:,0] |= depth[:,:,:,2] << 16
    depth = depth[:,:,:,0].astype(np.float32)
    depth /= e24



    stdev = stdev * stdev + np.std(depth)






with open(args.csv_file, 'r') as f:
    csv_f = csv.reader(f, delimiter=',')
    target = 0
    for line in csv_f:
        data[target].extend(line)
        target = 1 - target

data = list(zip(*data))

data2 = []
data3 = []
for line1, line2 in data:
    def process(line):
        line = line.replace('\n','')
        line = line.split(' ')
        line = line[1:-1]
        line = filter(lambda x: x != '', line)
        line = np.array([float(x) for x in line], dtype=np.float32)
        return line
    data2.append(process(line1))
    data3.append(process(line2))

data, data2 = data2, data3

data3, data4 = [], []
for l1, l2 in zip(data, data2):
    if len(l1) == 6 and len(l2) == 6:
        data3.append(l1)
        data4.append(l2)

data, data2 = data3, data4

data, data2 = data[-200:], data2[-200:]

data = np.array(data)
data2 = np.array(data2)

print "data mean: {:.4f}\ndata stdev: {:.4f}\ndata2 mean: {:.4f}\ndata2 stdev: {:.4f}".format(
        np.mean(data, axis=0), np.std(data, axis=0), np.mean(data2, axis=0), np.std(data2, axis=0))

delta = np.abs(data - data2)

print "delta mean: {:.4f}\ndelta stdev: {:.4f}".format(
        np.mean(delta, axis=0), np.std(delta, axis=0))

import pdb
pdb.set_trace()

print "done"




