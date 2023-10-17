#!/usr/bin/env python3

import argparse
import os


def kv_pair(string):
    splitter = string.split('=')
    if len(splitter) != 2:
        raise argparse.ArgumentTypeError(f"{string} is not a valid argument pair")
    return splitter[0], splitter[1]


ap = argparse.ArgumentParser()
ap.add_argument('pairs', metavar='NAME=VALUE', type=kv_pair, nargs='+')
args = ap.parse_args()
a_d = dict(args.pairs)

base_command = (f'sbatch -p {a_d["q"]}.q --gres=gpu:\"{a_d["gpu"]}\" -c 12 --mem=48GB'
                f' --job-name=vit --output={a_d["task"]}_{a_d["model"]}_{a_d["v"]}.out launch.sh ')

del a_d['q']
del a_d['gpu']

print(base_command + ' '.join([f'{k}={v}' for k, v in a_d.items()]))

os.system(base_command + ' '.join([f'{k}={v}' for k, v in a_d.items()]))