#!/bin/bash

python ./main.py \
--gpu_id 0 \
--train_dset_path C:\Users\9live\hm_code\open_data\office31\amazon\images \
--val_dset_path C:\Users\9live\hm_code\open_data\office31\dslr\images \
--RandomResizedCrop_train True \
--RandomResizedCrop_val True \
--RandomResizedCrop_size_train 245 \
--RandomResizedCrop_size_val 245 \
--CenterCrop_train True \
--CenterCrop_val True \
--CenterCrop_size_train 224 \
--CenterCrop_size_val 224 \
--Normalize True \
--Normalize_mean \
--Normalize_std \
--Resize_train False \
--Resize_size_train \
--Resize_val True \
--Resize_size_val 256 \
--train_batchsize 4 \
--val_batchsize 4  \
--imshow True \
--cv_imshow False \
--matplot_imshow False \