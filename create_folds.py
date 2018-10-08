from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import math
import numpy as np
from sklearn.model_selection import KFold

from utils import write_json

parser = argparse.ArgumentParser("Code to create splits in json form")
parser.add_argument('-d', '--dataset', type=str, required=True, help="path to h5 dataset (required)")
parser.add_argument('--save-dir', type=str, default='datasets', help="path to save output json file (default: 'datasets/')")
parser.add_argument('--save-name', type=str, default='splits', help="name to save as, excluding extension (default: 'splits')")
parser.add_argument('--num-splits', type=int, default=5, help="how many splits to generate (default: 5)")

args = parser.parse_args()

def create():
    print("==========\nArgs:{}\n==========".format(args))
    print("Goal: randomly folded data for {} times".format(args.num_splits))
    print("Loading dataset from {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    keys = np.array(dataset.keys())
    kf = KFold(n_splits=args.num_splits, shuffle=True, random_state=49)
    splits = []

    for train, test in kf.split(keys):
        splits.append({
            'train_keys': list(keys[train].astype(str)),
            'test_keys': list(keys[test].astype(str)),
        })

    saveto = osp.join(args.save_dir, args.save_name + '.json')
    write_json(splits, saveto)
    print("Splits saved to {}".format(saveto))

    dataset.close()

if __name__ == '__main__':
    create()