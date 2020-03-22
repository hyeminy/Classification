import argparse
import os
import settings

parser = argparse.ArgumentParser(description='Classification')

##################################################
# Env
##################################################
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=4)

##################################################
# Data
##################################################
parser.add_argument('--train_dset_path', type=str, default='C:\\Users\\9live\\hm_code\\open_code')
parser.add_argument('--val_dset_path', type=str, default='C:\\Users\\9live\\hm_code\\open_code')
#parser.add_argument('--test_dset_path', type=str, default='C:\\Users\\9live\\hm_code\\open_code')

parser.add_argument('--train_batchsize', type=int, default=32)
parser.add_argument('--val_batchsize', type=int, default=32)
#parser.add_argument('--test_batchsize', type=int, default=4)


parser.add_argument('--imshow', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--cv_imshow', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes'])) #parser.add_argument('--cv_imshow', type=bool, default=False) --> false 인식 못함
parser.add_argument('--matplot_imshow', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

##################################################
# Data Info
##################################################
#parser.add_argument('--class_num', type=int, default=10)

##################################################
# Data Pre-process from torch (torch에 있는 표현)
##################################################

#### Normalize
#parser.add_argument('--Normalize', type=bool, default=True)
parser.add_argument('--Normalize', default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']))

parser.add_argument('--Normalize_mean', type=float, default=[0.485, 0.456, 0.406] , help='ImageNet mean')
parser.add_argument('--Normalize_std', type=float, default=[0.229, 0.224, 0.225] , help='ImageNet std')


#### RandomResizedCrop
parser.add_argument('--RandomResizedCrop_train', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--RandomResizedCrop_size_train',type=int, default=256)

parser.add_argument('--RandomResizedCrop_val', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--RandomResizedCrop_size_val',type=int, default=256)

#### Centercrop

parser.add_argument('--CenterCrop_train', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--CenterCrop_size_train',type=int, default=224)

parser.add_argument('--CenterCrop_val', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--CenterCrop_size_val',type=int, default=224)

#### Resize
parser.add_argument('--Resize_train', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--Resize_size_train',type=int, default=224)

parser.add_argument('--Resize_val', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
parser.add_argument('--Resize_size_val',type=int, default=256)



'''
##################################################
# Network
##################################################
parser.add_argument('--net', type=str, default='resnet50', choices=['resnet50'])

##################################################
# Output
##################################################
parser.add_argument('--test_interval', type=int, default=500)
parser.add_argument('--output_dir', type=str, default='output')

##################################################
# Train
##################################################
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--num_iterations', type=int, default=10000)

parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD'])



"""LR 스케쥴러, optimizer에 대해서 더 업데이트 하기"""
'''
########################################################################################
'''
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

config={}
config['gpu_id'] = args.gpu_id
config['num_workers'] = args.num_workers

#config['data']={}
config['train_dset_path']=args.train_dset_path
config['val_dset_path']=args.val_dset_path
# config['test_dset_path']=args.test_dset_path
#
config['train_batchsize']=args.train_batchsize
config['val_batchsize']=args.val_batchsize
# config['test_batchsize']=args.test_batchsize
#
# config['class_num']=args.class_num

config['imshow']=args.imshow
config['cv_imshow']=args.cv_imshow
config['matplot_imshow']=args.matplot_imshow
####################################
# Pre-process
####################################

config['Normalize'] = args.Normalize
config['Normalize_mean'] = args.Normalize_mean
config['Normalize_std'] = args.Normalize_std

config['RandomResizedCrop_train']=args.RandomResizedCrop_train
config['RandomResizedCrop_size_train']=args.RandomResizedCrop_size_train

config['RandomResizedCrop_val']=args.RandomResizedCrop_train
config['RandomResizedCrop_size_val']=args.RandomResizedCrop_size_train

config['CenterCrop_train'] = args.CenterCrop_train
config['CenterCrop_size_train'] = args.CenterCrop_size_train

config['CenterCrop_val'] = args.CenterCrop_val
config['CenterCrop_size_val'] = args.CenterCrop_size_val

config['Resize_train']=args.Resize_train
config['Resize_size_train']=args.Resize_size_train

config['Resize_val']=args.Resize_val
config['Resize_size_val']=args.Resize_size_val

# config['Colorjitter'] = args.Colorjitter
# config['Colorjitter_brightness'] = args.brightness
# config['Colorjitter_contrast'] = args.contrast
# config['Colorjitter_saturation'] = args.saturation
# config['Colorjitter_hue'] = args.hue




# config['num_iterations'] = args.num_iterations
# config['test_interval'] = args.test_interval
# config['output_path'] = args.output_dir
# if not os.path.exists(config["output_path"]):
#     os.mkdir(config["output_path"])
# 
# config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w')
# if not os.path.exists(config["output_path"]):
#     os.mkdir(config["output_path"])
# 
# config['optimizer'] = {}
# config['optimizer']['type']=args.optimizer
# #config['optimizer']['optim_params']=
# 
# 
# config['lr']=args.lr
# 
# config['network']=args.net

#config['out_file'].write(str(config))
#config['out_file'].flush()
'''
#########################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = settings.gpu_id

config={}
config['gpu_id'] = settings.gpu_id
config['num_workers'] = settings.num_workers

#config['data']={}
config['train_dset_path']=settings.train_dset_path
config['val_dset_path']=settings.val_dset_path
# config['test_dset_path']=asettings.test_dset_path
#
config['train_batchsize']=settings.train_batchsize
config['val_batchsize']=settings.val_batchsize
# config['test_batchsize']=settings.test_batchsize
#
# config['class_num']=settings.class_num

config['imshow']=settings.imshow
config['cv_imshow']=settings.cv_imshow
config['matplot_imshow']=settings.matplot_imshow
####################################
# Pre-process
####################################

config['Normalize'] = settings.Normalize
config['Normalize_mean'] = settings.Normalize_mean
config['Normalize_std'] = settings.Normalize_std

config['RandomResizedCrop_train']=settings.RandomResizedCrop_train
config['RandomResizedCrop_size_train']=settings.RandomResizedCrop_size_train

config['RandomResizedCrop_val']=settings.RandomResizedCrop_train
config['RandomResizedCrop_size_val']=settings.RandomResizedCrop_size_train

config['CenterCrop_train'] = settings.CenterCrop_train
config['CenterCrop_size_train'] = settings.CenterCrop_size_train

config['CenterCrop_val'] = settings.CenterCrop_val
config['CenterCrop_size_val'] = settings.CenterCrop_size_val

config['Resize_train']=settings.Resize_train
config['Resize_size_train']=settings.Resize_size_train

config['Resize_val']=settings.Resize_val
config['Resize_size_val']=settings.Resize_size_val

# config['Colorjitter'] = settings.Colorjitter
# config['Colorjitter_brightness'] = settings.brightness
# config['Colorjitter_contrast'] = settings.contrast
# config['Colorjitter_saturation'] = settings.saturation
# config['Colorjitter_hue'] = settings.hue



'''
config['num_iterations'] = settings.num_iterations
config['test_interval'] = settings.test_interval
config['output_path'] = settings.output_dir
if not os.path.exists(config["output_path"]):
    os.mkdir(config["output_path"])

config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w')
if not os.path.exists(config["output_path"]):
    os.mkdir(config["output_path"])

config['optimizer'] = {}
config['optimizer']['type']=settings.optimizer
#config['optimizer']['optim_params']=


config['lr']=settings.lr

config['network']=settings.net
'''
#config['out_file'].write(str(config))
#config['out_file'].flush()
