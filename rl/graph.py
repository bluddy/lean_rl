#!/usr/bin/env python

import numpy as np
import os, sys, argparse
from os.path import abspath
from os.path import join as pjoin
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

''' Create graphs from experiment csv data '''

def load_info_file(file):
    files, labels = [], []
    with open(file, 'r') as f:
        csv_f = csv.reader(f, delimiter=',')
        for line in csv_f:
            if len(line) == 0:
                break
            files.append(line[0])
            labels.append(line[1])
    return files, labels

# CSV: t, r, q_avg, q_max, loss_avg, best_avg_r, last_learn_t, last_eval_t, succ1_pct, succ2_pct
def load_files(files, labels=None, tmax=None, div=50, info_file=None, max_value=False):
    if info_file is not None:
        files, labels = load_info_file(info_file)

    # Load files. Data is organized by types of data (e.g. different expeiments)
    data = []
    # Get lowest max t among files
    if not tmax:
        tmax = sys.maxsize
        for f in files:
            with open(f, 'r') as f:
                csv_f = csv.reader(f, delimiter=',')
                for line in csv_f:
                    t = int(line[0])
            if t < tmax:
                tmax = t

    for f in files:
        ts, rs, s1, s2 = [], [], [], []
        with open(f, 'r') as f:
            csv_f = csv.reader(f, delimiter=',')
            for line in csv_f:
                t = int(line[0])
                if t > tmax:
                    break
                if line[0][0] == '#':
                    continue
                ts.append(t)
                rs.append(float(line[1]))
                s1.append(float(line[8]))
                s2.append(float(line[9]))
        ts, rs, s1, s2 = (np.array(ts), np.array(rs), np.array(s1), np.array(s2))
        r_avg, _, r_low, r_high = utils.get_stats(rs, div)
        s1_avg, _, s1_low, s1_high = utils.get_stats(s1, div)
        s2_avg, _, s2_low, s2_high = utils.get_stats(s2, div)
        data.append({
            "t":ts,
            "r":rs,
            "s1":s1,
            "s2":s2,
            "r_avg":r_avg,
            "s1_avg":s1_avg,
            "s2_avg":s2_avg,
            "r_low":r_low,
            "s1_low":s1_low,
            "s2_low":s2_low,
            "r_high":r_high,
            "s1_high":s1_high,
            "s2_high":s2_high,
        })

    if max_value:
        # Check if we need to append last values at tmax
        do_append = False
        for d in data:
            if d['t'][-1] < tmax:
                d['t'] = np.append(d['t'], tmax)
                do_append = True
            for s in ['r', 's1', 's2', 'r_avg', 's1_avg', 's2_avg']:
                if do_append:
                    d[s] = np.append(d[s], 0.)
                utils.set_max_value_over_time(d[s])

    if labels is None:
        labels = [str(x) for x in range(len(files))]
    use_legend = len(files) > 1

    use_succ2 = False
    for d in data:
        if np.any(d["s2"] != 0.):
            use_succ2 = True

    # Plot R_avg
    fig = plt.figure()
    for d, label in zip(data, labels):
        length = len(d["r_avg"])
        plt.plot(d["t"][:length], d["r_avg"], label=label)
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if use_legend:
        plt.legend(frameon=True)
    plt.savefig('rewards_avg.png')

    # Plot R
    fig = plt.figure()
    for d, label in zip(data, labels):
        plt.plot(d["t"], d["r"], label=label)
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    if use_legend:
        plt.legend(frameon=True)
    plt.savefig('rewards.png')

    # Plot Success
    fig = plt.figure()
    for d, label in zip(data, labels):
        plt.plot(d["t"], d["s1"], label=label)
    plt.xlabel('steps')
    plt.ylabel('success')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(frameon=True)
    plt.savefig('success1.png')

    # Plot Average Success
    fig = plt.figure()
    for d, label in zip(data, labels):
        length = len(d["s1_avg"])
        plt.plot(d["t"][:length], d["s1_avg"], label=label)
    plt.xlabel('steps')
    plt.ylabel('success')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.legend(frameon=True)
    plt.savefig('success1_avg.png')

    if use_succ2:
        # Plot Success
        fig = plt.figure()
        for d, label in zip(data, labels):
            length = len(d["s2_avg"])
            plt.plot(d["t"], d["s2"], label=label)
        plt.xlabel('steps')
        plt.ylabel('success')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(frameon=True)
        plt.savefig('success2.png')

        # Plot Average Success
        fig = plt.figure()
        for d, label in zip(data, labels):
            length = len(d["s2_avg"])
            plt.plot(d["t"][:length], d["s2_avg"], label=label)
        plt.xlabel('steps')
        plt.ylabel('success')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(frameon=True)
        plt.savefig('success2_avg.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*', help='CSV files')
    parser.add_argument('--labels', nargs='+', help='Labels for files')
    parser.add_argument('--info', default=None, help='Info for labels, files')
    parser.add_argument('--tmax', default=None, type=int, help='Max time')
    parser.add_argument('--div', default=10, type=int, help='Mean points in graph')
    parser.add_argument('--max-value', default=False, action='store_true',
            help='Graph the max values achieved (e.g. for non-random envs)')
    args = parser.parse_args()

    load_files(args.files, tmax=args.tmax, div=args.div, labels=args.labels, info_file=args.info, max_value=args.max_value)

