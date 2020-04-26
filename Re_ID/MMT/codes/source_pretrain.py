from __future__ import print_function, absolute_import
import argparse
import json
import os
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

start_epoch = 0
best_mAP = 0

def _load_experiments_config_from_json(args, json_path, arg_parser):
    with open(json_path, 'r') as f:
        config = json.load(f)
    for config_name, config_val in config.items():
        if config_name in args.__dic__ and getattr(args, config_name) == arg_parser.get_default(config_name):
            setattr(args, config_name, config_val)
    print("Config at '{}' has been loaded".format(json_path))


def get_parsed_args(arg_parser: argparse.ArgumentParser):
    args = arg_parser.parse_args()
    if args.config_path:
        _load_experiments_config_from_json(args, args.config_path, arg_parser)
    return args


def main(args):
    if args.seed is not None:
        pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pre-training on the source domain")
    parser.add_argument('--config_path', type=str, default='', help='config json path')

    # data
    parser.add_argument('-ds', '--dataset_source', type=str, default='market1501', choices=['market1501'])
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc', choices=['dukemtmc'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50'])

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warmup_step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    parser.add_argument('--data_dir', type=str, metavar='PATH',default="")
    parser.add_argument('--logs_dir', type=str, metavar='PATH',default="")

    args = get_parsed_args(parser)
    args.out_file = open(os.path.join(args.log_dir, 'log.txt'), 'w')
    if not os.path.exists(args.exp_output):
        os.mkdir(args.exp_output)
    main(args)