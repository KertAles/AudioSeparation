# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 21:26:06 2022

@author: Kert PC
"""

from scipy.io.wavfile import read
import numpy as np
from os import listdir
from os.path import join


def read_from_file(path, target="mixture.wav") :
    song_data = []
    songs = [f for f in listdir(path)]
    
    for song in songs :
        song_wav = read(join(path, song, target))
        sample = song_wav[0]
        data = song_wav[1][:, 0]
        
        i = 0
        
        while i + sample < len(data) :
            song_data.append(data[i : i + sample])
            i += sample - (sample // 4)

    return np.array(song_data)


def get_set(path, mode="vocals") :
    gt = read_from_file(path, mode + ".wav")
    inpt = read_from_file(path, "mixture.wav")
    
    return inpt, gt



import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import math
import numpy as np
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, path):
        #self.signals, self.gt = get_set(path, 'vocals')
        self.path = path
        self.ids = [f for f in listdir(path)]
        
        self.length = 0
        self.indices = []
        self.sample_length = 16
        self.segments_per_song = 5
        self.prev_song = -1
        
        for idd in self.ids :
            song_wav = read(join(self.path, idd, 'mixture.wav'))
            samplerate = song_wav[0]
            
            self.indices.append(self.length)
            song_length = len(song_wav[1][:, 0])
            #self.length += math.ceil(song_length / (samplerate * self.sample_length))
            self.length += self.segments_per_song
            
        
        
    def __len__(self):
        return self.length
  
    def find_curr_song(self, idx) :
        curr_song = 0
        
        for i, index in enumerate(self.indices) :
            if index > idx :
                break
            curr_song = i
            
        return curr_song
        
    def load(self, idx) :
        
        """
        curr_song = self.find_curr_song(idx)
        
        if (curr_song != self.prev_song) :
            self.song_wav = read(join(self.path, self.ids[curr_song], 'mixture.wav'))
            self.gt_wav = read(join(self.path, self.ids[curr_song], 'vocals.wav'))
            self.prev_song = curr_song
        
        start_idx = self.indices[curr_song]
        sample = self.song_wav[0] * self.sample_length
        sample_idx = idx - start_idx
        
        signal = np.zeros(sample)
        gt = np.zeros(sample)
        
        end_idx = min(len(self.song_wav[1][:, 0]) - 1, sample * (sample_idx + 1))
        
        intermediate_signal = (np.array(self.song_wav[1][sample*sample_idx:end_idx, 0], dtype=float)
                               + np.array(self.song_wav[1][sample*sample_idx:end_idx, 1], dtype=float)) / 2
        signal[:len(intermediate_signal)] = intermediate_signal
        signal = np.expand_dims(signal / 32768.0, axis=0)
        
        intermediate_gt = (np.array(self.gt_wav[1][sample*sample_idx:end_idx, 0], dtype=float)
                            + np.array(self.gt_wav[1][sample*sample_idx:end_idx, 1], dtype=float)) / 2
        gt[:len(intermediate_gt)] = intermediate_gt
        gt = np.expand_dims(gt / 32768.0, axis=0)
        
        #gt = (np.array(gt_wav[1][sample*2:sample*3, 0], dtype=float) + np.array(gt_wav[1][sample*2:sample*3, 1], dtype=float))
        #gt = np.expand_dims(gt / 32768.0, axis=0)
        """
        
        
        curr_song = idx // self.segments_per_song
        curr_idx = idx % self.segments_per_song
        
        song_wav = read(join(self.path, self.ids[curr_song], 'mixture.wav'))
        gt_wav = read(join(self.path, self.ids[curr_song], 'vocals.wav'))
        
        sample = song_wav[0] * self.sample_length
        
        
        signal = (np.array(song_wav[1][sample*(curr_idx + 1):sample*(curr_idx + 2), 0], dtype=float)
                               + np.array(song_wav[1][sample*(curr_idx + 1):sample*(curr_idx + 2), 1], dtype=float)) / 2
        signal = np.expand_dims(signal / 32768.0, axis=0)
        
        
        gt = (np.array(gt_wav[1][sample*(curr_idx + 1):sample*(curr_idx + 2), 0], dtype=float)
                              + np.array(gt_wav[1][sample*(curr_idx + 1):sample*(curr_idx + 2), 1], dtype=float)) / 2
        gt = np.expand_dims(gt / 32768.0, axis=0)
        
        
        
        assert signal.size == gt.size
        
        return signal, gt
        
  
    def __getitem__(self, idx):
 
        #print('Loading data : ' + self.ids[idx])
        
        track, separation = self.load(idx)
        


        assert track.size == separation.size, \
            f'Track and separation {idx} should be the same size, but are {track.size} and {separation.size}'

        return {
            'track': torch.as_tensor(track.copy()).float().contiguous(),
            'separation': torch.as_tensor(separation.copy()).float().contiguous()
        }