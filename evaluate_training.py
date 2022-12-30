# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:54:49 2022

@author: Kert PC
"""

# The purpose of this file is to reconstruct the loss values during training,
# as the original file was tragically lost in an overwriting incident.



import logging
import sys
from pathlib import Path
from os import listdir
import os

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from read_data import get_set, BasicDataset

from evaluate import evaluate_with_loss
from unet.unet_model import UNet


dir_train = Path('./data/train/')
dir_test = Path('./data/test/')
dir_checkpoint2 = Path('./checkpoints_second_stage/')
dir_checkpoint1 = Path('./checkpoints_first_stage/')
f_l2 = open('l2_rec.txt', 'w')
f_loss = open('loss_rec.txt', 'w')



def reconstruct_net(net,
              device,
              val_loader
              ):

    criterion = nn.MSELoss()
    val_score, loss = evaluate_with_loss(net, val_loader, device, criterion)
    
    f_loss.write(str(loss))
    f_loss.write('\n')
    f_l2.write(str(val_score))
    f_l2.write('\n')

    logging.info('Validation MSE loss: {}'.format(loss))
    logging.info('Validation L2 norm: {}'.format(val_score))

            
            

if __name__ == '__main__':
    args = {'epochs' : 80,
            'batch_size' : 3,
            'lr' : 2e-5,
            'load' : True,
            'val' : 10.0,
            'classes' : 1,
            }

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    dataset = BasicDataset(dir_train)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    checkpoints1 = [os.path.join(dir_checkpoint1, f) for f in listdir(dir_checkpoint1)]
    checkpoints2 = [os.path.join(dir_checkpoint2, f) for f in listdir(dir_checkpoint2)]
    checkpoints = checkpoints1 + checkpoints2
    
    for checkpoint in checkpoints:
        logging.info('Evaluating checkpoint : ' + checkpoint)
        net = UNet(n_channels=1, n_classes=args['classes'], bilinear=False)
        net.load_state_dict(torch.load(checkpoint, map_location=device))
        net.to(device=device)
        net.eval()
        
        reconstruct_net(net, device, val_loader)
