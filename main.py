"""
@Author: Kang Ge
"""

import torch
import os
from torch.utils.data import random_split
import torch.utils.data as Data
from getdata import get_data_stored
from model import *
from train import *
from test import Test
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,default="D:/stead/", help='Your path where STEAD dataset is stored')
parser.add_argument('--conv_dim', type=list,default=[3,32,64,128], help='dimension of each convolution layer')
parser.add_argument('--projection_dim', type=int,default=128, help='the input dimension of transformer, the same as conv_dim[-1]')
parser.add_argument('--ff_size', type=int,default=256, help='middle dimension of mlp in transformer layer')
parser.add_argument('--num_predictions', type=int,default=4, help='Number of source parameters to be predicted. 4 for magnitude, epicentral distance, p travel time, depth')
parser.add_argument('--atten_type', type=str, default='FFT', help='"Attention" for using the standard transformer encoder, "FFT" for using the Fourier Transformer ')
parser.add_argument('--drop_rate', type=float, default=0.1)
parser.add_argument('--transformer_layers', type=int,default = 4, help='Number of transformer layers used')
parser.add_argument('--train_size', type=float,default = 0.75, help='The proportion of the training set')
parser.add_argument('--valid_size', type=float,default = 0.15, help='The proportion of the validation set')
parser.add_argument('--pretrained', default = None, help='If use pretrained checkpoint')
parser.add_argument('--patience', type=int,default=30, help='Early stopping is used if val_loss remain undecreased for "patience" epochs')
parser.add_argument('--patience_val', type=int,default=10, help='Reduce lr by a factor when the val_loss remain unc=decreased for "patience_val" epochs')
parser.add_argument('--batch_size_train', type=int,default=256, help='Bs for training set')
parser.add_argument('--batch_size_valid', type=int,default=128, help='Bs for validation set')
parser.add_argument('--maximun_epoch', type=int,default=200)
parser.add_argument('--initial_lr', type=float, default= 0.001)
parser.add_argument('--factor', type=float, default= 0.316, help='Scaling factor for the decay of learning rate')

args = parser.parse_args()


if __name__ == "__main__":

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)

    folder =  os.path.dirname(os.path.abspath(__file__)).replace('\\','/') +'/Results'
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_path = args.data_path 

    torch.cuda.set_device(0)
    print("cuda:")
    print(torch.cuda.current_device())

    dataset = get_data_stored(data_path)

    model=MFTnet(seq_len=24,conv_dim=args.conv_dim,projection_dim=args.projection_dim,transformer_layers=args.transformer_layers,
                    ff_size=args.ff_size,drop_rate=args.drop_rate,num_predictions=args.num_predictions,atten_type=args.atten_type)

    train_size = int(args.train_size * len(dataset))
    valid_size = int(args.valid_size * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_and_test_dataset = random_split(dataset, [train_size, valid_size + test_size])
    valid_dataset, test_dataset = random_split(valid_and_test_dataset, [valid_size, test_size])

    Train(train_dataset,valid_dataset,model,folder,args)

    Test(test_dataset,model,folder)

