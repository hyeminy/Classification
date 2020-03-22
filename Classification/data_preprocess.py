from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from transforms_custom import *


def data_transform(config):
    data_transforms = {}
    data_transforms["train"] = image_train_transform(config)
    """
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    """
    data_transforms["val"] = image_val_transform(config)
    return data_transforms


def image_train_transform(config): # 전처리 종류

    train_compose = []

    #################################
    # Custom transforms
    #################################
    #if config['Resize_custom']:
        #train_compose.append(transforms.Resize)

    #################################
    # torch transforms
    #################################

    if config['RandomResizedCrop_train']:
        """
        CLASS torchvision.transforms.RandomResizedCrop(size, 
                                                      scale=(0.08, 1.0), 
                                                      ratio=(0.75, 1.3333333333333333), 
                                                      interpolation=2)
                                                      
        <Description>                               
        Crop the given PIL Image to random size and aspect ratio.
        A crop of random size (default: of 0.08 to 1.0) of the original size 
        and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. 
        This crop is finally resized to given size. 
        This is popularly used to train the Inception networks.
        
        <Parametetrs>
        size – expected output size of each edge
        scale – range of size of the origin size cropped
        ratio – range of aspect ratio of the origin aspect ratio cropped
        interpolation – Default: PIL.Image.BILINEAR                           
        """
        train_compose.append(transforms.RandomResizedCrop(config['RandomResizedCrop_size_train']))

    if config['CenterCrop_train']:
        """
        CLASS torchvision.transforms.CenterCrop(size)
        
        <Description>
        Crops the given PIL Image at the center.
        """
        train_compose.append(transforms.CenterCrop(config['CenterCrop_size_train']))

    if config['Resize_train']:
        train_compose.append(transforms.CenterCrop(config['Resize_size_train']))

    train_compose.append(transforms.ToTensor())

    if config['Normalize']:
        normalize = transforms.Normalize(mean = config['Normalize_mean'], std = config['Normalize_std'])
        train_compose.append(normalize)

    return transforms.Compose(train_compose)



def image_val_transform(config):
    val_compose = []

    #################################
    # Custom transforms
    #################################
    # if config['Resize_custom']:
    # train_compose.append(transforms.Resize)

    #################################
    # torch transforms
    #################################
    if config['RandomResizedCrop_val']:
        val_compose.append(transforms.RandomResizedCrop(config['RandomResizedCrop_size_val']))

    if config['Resize_val']:
        val_compose.append(transforms.CenterCrop(config['Resize_size_val']))


    if config['CenterCrop_val']:
        """
        CLASS torchvision.transforms.CenterCrop(size)

        <Description>
        Crops the given PIL Image at the center.
        """
        val_compose.append(transforms.CenterCrop(config['CenterCrop_size_val']))

    val_compose.append(transforms.ToTensor())

    if config['Normalize']:
        normalize = transforms.Normalize(mean=config['Normalize_mean'], std=config['Normalize_std'])
        val_compose.append(normalize)

    return transforms.Compose(val_compose)




def data_loaders(config, image_datasets):
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],\
                                                       batch_size=config['train_batchsize'],\
                                                       shuffle=True,\
                                                       num_workers=config['num_workers'],\
                                                       drop_last=False)

    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],\
                                                     batch_size=config['val_batchsize'],\
                                                     shuffle=False,\
                                                     num_workers=config['num_workers'],\
                                                     drop_last=False)

    return dataloaders
