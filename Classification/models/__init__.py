from collections import OrderedDict
from abc import ABC

import torch
from torch import nn as nn

import torch
from models.resnet import create_resnet_lower

def create_feature_extractor(args):
    if args.feature_extractor_model == 'resnet50': # 원본 resnet50
        encoder = create_resnet_lower(args.feature_extractor_model, args.feature_extractor_pretrain)

    elif args.feature_extractor_model == 'resnet50_dta':
        encoder = create_resnet_lower(args.feature_extractor_model, args.feature_extractor_pretrain)