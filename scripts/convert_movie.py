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

    if args.mode == 'mono':
        tgt_width_end = height
        tgt_start = 0
        suffix = 'm'
    elif args.mode == 'stereo':
        tgt_width_end = height * 2
        tgt_start = 0
        suffix = 'st'
    elif args.mode == 'depth':
        tgt_width_end = height * 2 * 3
        tgt_start = height 
        suffix = 'd'

    tgt_width = tgt_width_end - tgt_start

    # Check if resize
    if tgt_width > args.out_width:
        resize = True
        ratio = float(args.out_width) / tgt_width
        resize_height = int(args.out_height * ratio)
        x_off = 0
        y_off = (args.out_height - resize_height) / 2
    else:
        # Create black image
        x_off = (args.out_width - tgt_width) / 2
        y_off = (args.out_height - height) / 2

    bg = np.zeros((args.out_height, args.out_width, 3), dtype=np.uint8)

    # Reshape and save images
    for i, image in enumerate(images):
        bg2 = bg.copy()
        if resize:
            img = image[:,tgt_start:tgt_width_end,:]
            img = Image.fromarray(img)
            img = img.resize((args.out_width, resize_height))
            img = np.asarray(img)
            bg2[y_off:-y_off, :, :] = img
        else:
            bg2[y_off:-y_off, x_off:-x_off, :] = image[:,tgt_start:tgt_width_end,:]
        img = Image.fromarray(bg2)
        img.save('_img{:05d}.png'.format(i))

    out_file = os.path.splitext(args.file)[0] + '_' + suffix + '.mp4'
    if os.path.isfile(out_file):
        os.remove(out_file)

    pattern = '_img*.png'
    filelist = glob.glob(pattern)
    if len(filelist) > 0:
        cmd = 'cat {} | ffmpeg -f image2pipe -r 30 -vcodec png -i - -f lavfi -i anullsrc -vcodec h264 -preset ultrafast -pix_fmt yuv420p -g 1 -crf 1 -shortest {}'.format(pattern, out_file)
        os.system(cmd)
        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='movie file to load')
    parser.add_argument('--mode', help='mono|stereo|depth', default='mono')
    parser.add_argument('--size', default=448, type=int)
    parser.add_argument('--width-mult', type=int, default=3)
    parser.add_argument('--out-width', type=int, default=720)
    parser.add_argument('--out-height', type=int, default=486)
    args = parser.parse_args()
    run(args)

