# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

import torch
from torch.backends import cudnn
from evaluations import extract_features, pairwise_distance
import os
import numpy as np
from utils import to_numpy
from torch.nn import functional as F
import torchvision.transforms as transforms
from ImageFolder import *
from utils import *
from sklearn.metrics.pairwise import euclidean_distances
import random
from CIFAR100 import CIFAR100
import pdb


#from tensorboardX import SummaryWriter
#writer = SummaryWriter('logs')

def displacement(Y1, Y2, embedding_old, sigma):
    DY = Y2-Y1
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1])-np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1]))**2, axis=2)
    W = np.exp(-distance/(2*sigma ** 2))  # +1e-5
    W_norm = W/np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
                          1, 1, DY.shape[1]])*np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return displacement


cudnn.benchmark = True
parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('-data', type=str, default='cub')
parser.add_argument('-r', type=str, default='model.pkl', metavar='PATH')
parser.add_argument("-gpu", type=str, default='0', help='which gpu to choose')
parser.add_argument('-seed', default=1993, type=int, metavar='N',
                    help='seeds for training process')
parser.add_argument("-method", type=str, default='no', help='Choose FT or SC')
parser.add_argument('-mapping_test', help='Print more data',
                    action='store_true')
parser.add_argument('-sigma_test', default=0, type=float, help='sigma_test')
parser.add_argument('-real_mean', help='Print more data', action='store_true')
parser.add_argument('-epochs', default=600, type=int,
                    metavar='N', help='epochs for training process')
parser.add_argument('-exp', type=str, default='exp1',
                    help="learning rate of new parameters")
parser.add_argument('-task', default=1, type=int, help='task')
parser.add_argument('-base', default=50, type=int, help='task')


args = parser.parse_args()
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models = []


for i in os.listdir(args.r):
    print(i)
    
