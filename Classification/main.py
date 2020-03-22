from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, models, transforms

from options import config
import data_preprocess as prep
import os
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from data_analysis import *

import mymodels
import pprint




def main(config):
    ###########################################
    # Env
    ###########################################
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    print('Device : ', device)

    ###########################################
    # DATA load
    ###########################################

    data_transforms = prep.data_transform(config)

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(config['train_dset_path'], data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(config['val_dset_path'], data_transforms['val'])

    dataloaders = prep.data_loaders(config, image_datasets)

    dataset_sizes = {
        'train' : len(image_datasets['train']),
        'val' : len(image_datasets['val'])
    }
    print('Train_size : ', dataset_sizes['train'])
    print('Val_size : ', dataset_sizes['val'])

    class_names = image_datasets['train'].classes
    print('Class names : ', class_names)


    if config['imshow']:
        inputs, classes = next(iter(dataloaders['train'])) # one batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])





    ###########################################
    # Network
    ###########################################

    # networks={'resnet50' : mymodels.resnet50}
    #
    # model_ft = networks[config['network']](pretrained=True) #mymodels.py에 있는 함수 resnet50(pretrained=True) 호출
    # #print('------------------------------------------------')
    # #print(model_ft)
    # #print('------------------------------------------------')
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
    #
    # optimizers = {'SGD' : optim.SGD}
    #
    # optimizer_ft = optimizers[config['optimizer']['type']](model_ft.parameters(), lr = config['lr'], momentum = 0.9)
    #
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    # print('Done')





if __name__ == "__main__":
    if config['matplot_imshow']:
        print('here1')
        matplot_img_visualize(config)
    if config['cv_imshow']:
        print('here2')
        cv_img_visualize(config)
    pp = pprint.PrettyPrinter(width=20, indent=4)
    pp.pprint(config)
    main(config)