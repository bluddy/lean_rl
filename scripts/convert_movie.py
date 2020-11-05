#!/usr/bin/env python
''' Convet movies
'''

import os
import argparse
import numpy as np
import subprocess as sp
from PIL import Image
import glob

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

    if args.out == 'mono':
        tgt_width = height

    # Create black image
    x_off = (args.out_width - tgt_width) / 2
    y_off = (args.out_height - height) / 2
    bg = np.zeros((args.out_height, args.out_width, 3), dtype=np.uint8)

    # Reshape and save images
    if args.out == 'mono':
        for i, image in enumerate(images):
            img = bg.copy()
            img[y_off:-y_off, x_off:-x_off, :] = image[:,0:tgt_width,:]
            #img = image[:,0:tgt_width,:]
            img = Image.fromarray(img)
            img.save('_img{:05d}.png'.format(i))

    out_file = os.path.splitext(args.file)[0] + '_out.mp4'
    out_file2 = os.path.splitext(args.file)[0] + '_out2.mp4'
    if os.path.isfile(out_file):
        os.remove(out_file)

    pattern = '_img*.png'
    filelist = glob.glob(pattern)
    if len(filelist) > 0:
        cmd = 'cat {} | ffmpeg -f image2pipe -r 30 -vcodec png -i - -f lavfi -i anullsrc -vcodec h264 -preset ultrafast -pix_fmt yuv420p -g 1 -crf 1 -shortest {}'.format(pattern, out_file)
        os.system(cmd)
        #cmd = 'ffmpeg -i {} -f lavfi -i anullsrc -c:v copy -c:a aac -shortest {}'.format(out_file2, out_file)
        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='movie file to load')
    parser.add_argument('--out', help='mono|stereo|depth', default='mono')
    parser.add_argument('--size', default=448, type=int)
    parser.add_argument('--width-mult', type=int, default=3)
    parser.add_argument('--out-width', type=int, default=720)
    parser.add_argument('--out-height', type=int, default=486)
    args = parser.parse_args()
    run(args)

