# -*- coding: utf-8 -*-
"""
utils
some useful functions for training
"""
import random
import numpy as np
import torch
import os
import argparse
import yaml

def set_random(seed_id=1234):
    #set random seed for reproduce
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)
        
        
def get_args():
    parser = argparse.ArgumentParser() #can also add description
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--max_epoch', type=int,  default=405, help='maximum epoch number to train')
    parser.add_argument('--warmup_epoch',type=int,default=5,help='warmup lr epochs')
    parser.add_argument('--log_dir',type=str,default='../log/exp0304',help='log dir')
    parser.add_argument('--num_class',type=int,default=3,help='numer of class')
    parser.add_argument('--in_channels',type=int,default=4,help='number of modality')
    parser.add_argument('--data_dir',type=str,default='../data',help='dataset path')
    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
    parser.add_argument('--drop_rate',type=float,default=0,help='dropout rate')
    parser.add_argument('--num_filters',type=int,default=16,help='base num filters')
    #you can add more
    args = parser.parse_args()
    return args


def get_config(config):
    """load yaml file """
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)

from torch.nn import init
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    #print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>






