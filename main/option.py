import argparse

parser = argparse.ArgumentParser(description='Ours Model')

parser.add_argument('--model_name', type=str,
                    help='Choose the type of model to train or test')

parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--GPU_id', type=str, default='0',
                    help='Id of GPUs')
parser.add_argument('--seed', type=int, default=2025,
                    help='random seed')

parser.add_argument('--dir_data', type=str, default='D:/Code/PythonProject/DN/dataset/',
                    help='dataset directory')
parser.add_argument('--train_dataset', type=str, default="CBSD500",
                    help='Train dataset name')
parser.add_argument('--test_dataset', type=str, default="CBSD68",
                    help='Test dataset name')
parser.add_argument('--save_base', type=str, default='D:/Code/PythonProject/DN/',
                    help='save the value of loss per epoch')
parser.add_argument('--dir_loss', type=str, default='result/loss',
                    help='save the value of loss per epoch')
parser.add_argument('--dir_model', type=str, default='result/models/',
                    help='the model is saved to here')
parser.add_argument('--dir_state', type=str, default='result/state/',
                    help='the state is saved to here')
parser.add_argument('--dir_test_img', type=str, default='result/img/',
                    help='save the result of test img')
parser.add_argument('--pretrain', type=str, default='',
                    help='The file name of  pre_train model')

parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='the state is saved to here')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--n_pat_per_image', type=int, default=64,
                    help='a image produce n patches')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--aug_plus', action='store_true',
                    help='If use the data aug_plus')
parser.add_argument('--crop_batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--loss_func', type=str, default='l2',
                    help='choose the loss function')

parser.add_argument('--num_blocks', type=int, default=4,
                    help='number of multi-scale blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--sigma', type=float, default=25,
                    help='sigma == 100 means blind, sigma == 200 means realnoise')
parser.add_argument('--mode', type=str, default='train',
                    help='Choose to train or test or inference')
parser.add_argument('--model_file_name', type=str, default='',
                    help='load the mode_file_name')
parser.add_argument('--flag', type=int, default=0,
                    help='Choose the phase of experiment, 0 represent no experiment ')

args, unparsed = parser.parse_known_args()

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

