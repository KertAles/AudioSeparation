# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 20:45:18 2022

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

from torchmetrics.functional.audio import signal_distortion_ratio


dir_train = Path('./data/train/')
dir_test = Path('./data/test/')
dir_results = Path('./results/')
dir_checkpoint2 = Path('./checkpoints_second_stage/')
dir_checkpoint1 = Path('./checkpoints_first_stage/')
dir_spleeter = Path('./data/spleeter/')
dir_demucs = Path('./data/htdemucs/')



def get_original(track, length):
    signal = read_track(os.path.join(dir_test, track, 'vocals.wav'), True)
    signal = signal[:length]
    
    return signal
    
def read_track(path, stereo=False) :
    song_wav = read(path)
    if stereo:
        signal = ((np.array(song_wav[1][:, 0], dtype=float)
                + np.array(song_wav[1][:, 1], dtype=float)) / 2)
    else :
        signal = song_wav[1]
    
    return signal
    

if __name__ == '__main__':
    args = {'epochs' : 80,
            'batch_size' : 3,
            'lr' : 2e-5,
            'load' : True,
            'val' : 10.0,
            'classes' : 1,
            }
    
    tracks_sep = listdir(dir_demucs)
    tracks = listdir(dir_test)
    sdr_sum = 0
    
    for i,track in enumerate(tracks) :
        separation = read_track(os.path.join(dir_demucs, tracks_sep[i], 'vocals.wav'), True)
        original = get_original(track, len(separation))
        
        sep_tensor = torch.as_tensor(separation.copy()).float().contiguous()
        org_tensor = torch.as_tensor(original.copy()).float().contiguous()
        
        sdr = signal_distortion_ratio(sep_tensor, org_tensor)
        sdr_sum += sdr.item()
        
    print(sdr_sum / len(tracks))
        
        



