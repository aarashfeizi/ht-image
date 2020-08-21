import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
import argparse


def get_pos_and_negs(out_path):
    margin = float(out_path[out_path.find('margin'):][:out_path[out_path.find('margin'):].find('-')][6:])

    all_pos = []
    all_neg = []

    pos = []
    neg = []
    with open(out_path) as f:
        for line in f:
            if ' neg ' in line:
                neg.extend(line[(line.index('[') + 1):line.index(']')].split(', '))
            if 'pos' in line:
                pos.extend(line[(line.index('[') + 1):line.index(']')].split(', '))
            if '******************************' in line:
                all_neg.append(np.array(neg, dtype=np.float))
                all_pos.append(np.array(pos, dtype=np.float))
                pos = []
                neg = []
            if 'Save path:' in line:
                save_path = line.split('Save path: ')[1].strip()

    return all_pos, all_neg, save_path, margin


def plot_histograms(all_pos, all_neg, save_path, margin):
    save_path = os.path.join(save_path, 'histograms/')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(len(all_pos)):
        epoch = i + 1
        max_val = max(all_pos[epoch - 1].max(), all_neg[epoch - 1].max())
        min_val = min(all_pos[epoch - 1].min(), all_neg[epoch - 1].min())
        mean_diff = all_neg[epoch - 1].mean() - all_pos[epoch - 1].mean()
        mode_diff = mode(all_neg[epoch - 1], axis=None).mode[0] - mode(all_pos[epoch - 1], axis=None).mode[0]
        ret = plt.hist([all_pos[epoch - 1], all_neg[epoch - 1]], bins=1000, range=[min_val - 0.1, max_val + 0.1], alpha=0.3,
                 label=['pos', 'neg'])
        plt.legend(loc='upper right')

        plt.title(f'Epoch {epoch}')

        max_y = max(max(ret[0][0]), max(ret[0][1])) - int(max(max(ret[0][0]), max(ret[0][1])) / 13)

        plt.text(min_val - 0.1 + 0.01, max_y, f'Mean Diff = {mean_diff}\nMode Diff = {mode_diff}\nMargin = {margin}',
                 horizontalalignment='left',
                 verticalalignment='bottom')
        plt.savefig(os.path.join(save_path, f'epoch{epoch}.png'))
        plt.clf()
        print(f'Epoch {epoch} done')
        # plt.show()


parser = argparse.ArgumentParser()

parser.add_argument('-s', '--save_path', default='')

args = parser.parse_args()

all_pos, all_neg, save_path, margin = get_pos_and_negs(args.save_path)

plot_histograms(all_pos, all_neg, save_path, margin)
