import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from torchvision import transforms as transforms
from torchvision import datasets
import os

def transform_factory(transform_type, is_train=True, **kwargs): # 이 부분 나중에 transform 최적화 찾아주는 코드 추가하기, 데이터 normalize 값 찾아 주는 부분 추가하기
    if is_train:
        transform = {'source': transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.75, 1.33)),\
                                                  transforms.RandomHorizontalFlip(),\
                                                  transforms.ToTensor(),\
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                     'target' : transforms.Compose([transforms.Resize((224, 224)),\
                                                             transforms.RandomHorizontalFlip(),\
                                                             transforms.ToTensor(),\
                                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        }
    else: # val 할 때
        transform = { 'target' : transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

    return transform

def dataloaders_factory(args):
    source_train_transform = transform_factory(args.transform_type, is_train=True)['source'] #나중에 args.transformtype에 base,최적화방법 찾는것 추가하기
    target_train_transform = transform_factory(args.transform_type, is_train=True)['target']

    #train_transform = transform_factory(args.transform_type, is_train=True)

    val_transform = transform_factory(args.transform_type, is_train=False)

    # test용 코드는 나중에 짜기

    train_datasets = {}
    source_data_dir = args.source_dataset_dir
    target_data_dir = args.target_dataset_dir
    train_datasets['source'] = datasets.ImageFolder(os.path.join(source_data_dir), transform=source_train_transform)
    train_datasets['target'] = datasets.ImageFolder(os.path.join(target_data_dir), transform=target_train_transform)

    # print(len(dsets['source'])) 데이터 set 길이 확인 하는 코드 추가하기

    train_dataloaders = {}
    train_source_bs = args.source_batch_size
    train_target_bs = args.target_batch_size
    train_dataloaders['source'] = DataLoader(train_datasets["source"], batch_size=train_source_bs, shuffle=True, num_workers=4, drop_last=False)
    train_dataloaders['target'] = DataLoader(train_datasets["target"], batch_size=train_target_bs, shuffle=True, num_workers=4, drop_last=False)

    val_datasets = {}
    val_target_bs = args.val_batch_size
    val_datasets['target'] = datasets.ImageFolder(os.path.join(target_data_dir), transform=val_transform)

    val_dataloaders = {}
    train_source_bs = args.source_batch_size
    val_bs = args.val_batch_size
    val_dataloaders['target'] = DataLoader(val_datasets["target"], batch_size=val_target_bs, shuffle=False,num_workers=4, drop_last=False)


    return {'train': train_dataloaders, 'val':val_dataloaders}