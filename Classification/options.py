import argparse

parser = argparse.ArgumentParser(description='Options for UDA')


#########################
# 환경 Setting
#########################
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs')
parser.add_argument('--device_idx', type=str, default='0', help='Gpu idx')


#########################
# Experiment Logging Settings
#########################
parser.add_argument('--experiment_dir', type=str, default='', help='Experiment save directory')
parser.add_argument('--experiment_description', type=str, default='svhn_mnist', help='Experiment description')
parser.add_argument('--checkpoint_period', type=int, default=1, help='epoch / checkpoint_period = checkpoint num')

#########################
# Data load Settings
#########################
parser.add_argument('--source_dataset_dir', type=str, default='/DATA/DA/open_data/office31/amazon/images')
parser.add_argument('--target_dataset_dir', type=str, default='/DATA/DA/open_data/office31/webcam/images')

#########################
# Data 전처리 Settings
#########################
parser.add_argument('--transform_type', type=str, default='visda_standard', help='Transform type')
parser.add_argument('--num_classes', type=int, default=12)

#########################
# Training과정 Settings
#########################
parser.add_argument('--train_mode', type=str, default='dta', choices=['dta', 'source_only'],
                    help='Train mode')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--epoch', type=int, default=80, help='epoch (default: 100)')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization lambda (default: 0)')
parser.add_argument('--decay_step', type=int, default=15, help='num epochs for decaying learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay gamma')
parser.add_argument('--source_batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--target_batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--val_batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer')

#########################
# Test과정 Settings
#########################
parser.add_argument('--log_period_as_iter', type=int, default=10000, help='num iter')
parser.add_argument('--validation_period_as_iter', type=int, default=30000, help='validation period in iterations')
parser.add_argument('--test', type=bool, default=False, help='is Test')


#########################
# 모델 Settings
#########################
parser.add_argument('--classifier_ckpt_path', type=str, default='', help='Domain Classifier Checkpoint Path')
parser.add_argument('--encoder_ckpt_path', type=str, default='', help='Encoder Checkpoint Path')
parser.add_argument('--pretrain', type=str, default='',
                    choices=['class_classifier', 'domain_classifier', ''], help='Pretrain mode')
#parser.add_argument('--model', type=str, default='resnet50', help='Model: resnet50 | resnet101')
parser.add_argument('--feature_extractor_model', type=str, default='resnet50', help='Model: resnet50 | resnet50_dta' )
parser.add_argument('--feature_extractor_pretrain', type=bool, default=True)
#########################
# Loss Settings
#########################
parser.add_argument('--source_consistency_loss', type=str, default='l2', choices=['l1', 'l2', 'kld'],
                    help='Source Consistency Loss')



def get_parsed_args(arg_parser: argparse.ArgumentParser):
    args = arg_parser.parse_args()
    return args