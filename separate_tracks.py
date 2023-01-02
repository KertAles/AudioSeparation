# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 19:34:47 2022

@author: Kert PC
"""

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

from scipy.io.wavfile import read
import numpy as np
from os import listdir
from os.path import join
from scipy.io.wavfile import write



dir_train = Path('./data/train/')
dir_test = Path('./data/test/')
dir_results = Path('./results/')
dir_checkpoint2 = Path('./checkpoints_second_stage/')
dir_checkpoint1 = Path('./checkpoints_first_stage/')



def separate_track(net,
              device,
              track
              ):
    separations = []
    
    song_wav = read(join(track, 'mixture.wav'))
    sample = song_wav[0] * 16
    song_length = len(song_wav[1][:, 0])
    i = 0
    
    signal = ((np.array(song_wav[1][:, 0], dtype=float)
            + np.array(song_wav[1][:, 1], dtype=float)) / 2) / 32768.0
    #signal = np.expand_dims(signal , axis=0)
    
    while (i+1) * sample < song_length :
        
        snippet = np.expand_dims(signal[i*sample : (i+1)*sample] , axis=0)
        snippet = np.expand_dims(snippet , axis=0)
        snippet_tensor = torch.as_tensor(snippet.copy()).float().contiguous()
        
        snippet_tensor = snippet_tensor.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred_separation = net(snippet_tensor)
            separations.append(pred_separation.cpu().numpy() * 32768.0)
        
        i += 1
    
    separation = np.concatenate(separations, axis=2)[0, 0, :]

    
    return separation
    


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

    checkpoint = os.path.join(dir_checkpoint2, 'checkpoint_epoch80.pth')
    
    logging.info('Evaluating checkpoint : ' + checkpoint)
    net = UNet(n_channels=1, n_classes=args['classes'], bilinear=False)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net.to(device=device)
    net.eval()
    
    tracks = listdir(dir_test)
    
    for track in tracks :
        separation = separate_track(net, device, os.path.join(dir_test, track))
        write(os.path.join(dir_results, track + '.wav'), 44100, separation.astype(np.int16))
        
        
