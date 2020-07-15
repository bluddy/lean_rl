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

# CSV: t, r, q_avg, q_max, loss_avg, best_avg_r, last_learn_t, last_eval_t, succ1_pct, succ2_pct
def load_files(files, tmax):
    # Load files
    data = []
    for f in files:
        ts, rs, s1, s2 = [], [], [], []
        with open(f, 'r') as f:
            csv_f = csv.reader(f, delimiter=',')
            for line in csv_f:
                t = int(line[0])
                if t > t_max:
                    break
                ts.append(t)
                rs.append(float(line[1]))
                s1.append(float(line[8]))
                s2.append(float(line[9]))
        ts, rs, s1, s2 = (np.array(ts), np.array(rs), np.array(s1), np.array(s2))
        r_avg, _, r_low, r_high = utils.get_stats(rs)
        s1_avg, _, s1_low, s1_high = utils.get_stats(s1)
        s2_avg, _, s2_low, s2_high = utils.get_stats(s2)
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

    # Plot R_avg
    fig = plt.figure()
    for d in data:
        length = len(d["r_avg")
        plt.plot(d["t"][:length], d["r_avg"], label='Average Rewards')
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.savefig('rewards_avg.png'))

    # Plot R
    fig = plt.figure()
    for d in data:
        plt.plot(d["t"], d["r"], label='Rewards')
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.savefig('rewards.png'))

    # Plot Success
    fig = plt.figure()
    for d in data:
        use_succ2 = np.any(d["s2"] != 0.)
        if use_succ2:
            plt.plot(d["t"], d["s1"] + d["s2"], label='State 2')
        plt.plot(d["t"], d["s1"], label='State 1')
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.savefig('success.png'))

    # Plot Success
    fig = plt.figure()
    for d in data:
        length = len(d["s1_avg"])
        use_succ2 = np.any(d["s2_avg"] != 0.)
        if use_succ2:
            plt.plot(d["t"][:length], d["s1"] + d["s2"], label='State 2')
        plt.plot(d["t"][:length], d["s1"], label='State 1')
        #plt.fill_between(total_times_nd[:length], r_low, r_high, alpha=0.4)
    plt.savefig('success_avg.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', required=True, nargs='+', help='CSV files')
    parser.add_argument('tmax', default=None, type=int, help='Max time')
    args = parser.parse_args()

    load_files(args.files, tmax=args.tmax)

