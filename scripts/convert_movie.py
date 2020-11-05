#!/usr/bin/env python
''' Convet movies
'''

import os
import argparse
import numpy as np
import subprocess as sp

def run(args):
    height = args.size
    width = height * args.width_mult

    # Open video file
    command = [ 'ffmpeg',
        '-loglevel', '8',
        '-i', args.file,
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo', '-'
        ]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    images = []
    while True:
        raw_image = pipe.stdout.read(width * height * 3)
        if len(raw_image) == 0:
            break
        image = np.fromstring(raw_image, dtype='uint8')
        image = image.reshape((height, width, 3))
        images.append(image)
    pipe.stdout.close()

    # Reshape
    images2 = []
    if args.out == 'mono':
        tgt_width = height
        for image in images:
            img = image[:,0:tgt_width,:]
            images2.append(img)

    tgt_file = os.path.splitext(args.file)[0] + '_out.mp4' 
    if os.path.isfile(tgt_file):
        os.remove(tgt_file)


    # Dump file
    command = [ 'ffmpeg',
        #'-loglevel', '8',
        '-s', '{}x{}'.format(tgt_width, height),
        '-f', 'image2pipe',
        '-r', '5',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-i', '-',
        #'-an',
        '-vcodec', 'libx264',
        '-preset', 'veryslow',
        tgt_file
        ]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    for img in images2:
        import pdb
        pdb.set_trace()
        #img = img.copy(order='C')
        s = img.tostring()
        pipe.stdin.write(s)

    pipe.stdin.close()
    pipe.wait()

    if pipe.returncode != 0:
        raise sp.CalledProcessError(pipe.returncode, command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='movie file to load')
    parser.add_argument('--out', help='mono|stereo|depth', default='mono')
    parser.add_argument('--size', default=448, type=int)
    parser.add_argument('--width_mult', type=int, default=3)
    args = parser.parse_args()
    run(args)

