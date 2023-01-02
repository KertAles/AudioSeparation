# -*- coding: utf-8 -*-
 
"""
Created on Sun Dec  4 11:44:25 2022

@author: Kert PC
"""

import logging
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import torch
import torch.nn as nn
import torch.nn.functional as F
#import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from read_data import get_set, BasicDataset

from evaluate import evaluate
from unet.unet_model import UNet


dir_train = Path('./data/train/')
dir_test = Path('./data/test/')
dir_checkpoint = Path('./checkpoints/')
f_l2 = open('l2.txt', 'w')
f_loss = open('loss.txt', 'w')


def test_net(net,
             device) :
    
    net.eval()
    dataset = BasicDataset(dir_test)
    
    loader_args = dict(batch_size=1, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=True, **loader_args)
    
    for batch in tqdm(test_loader, total=len(test_loader), desc='Testing', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        

        with torch.no_grad():
            mask_pred = net(image)
            
            separate_np = mask_pred.cpu().numpy()
            separate_np = separate_np * 32768
            
            samplerate = 44100
            wavfile.write("example.wav", samplerate, separate_np.astype(np.int16))
            


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True):
    # 1. Create dataset
    #dataset = get_set(dir_train)
    dataset = BasicDataset(dir_train)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    #experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    #experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                              val_percent=val_percent, save_checkpoint=save_checkpoint))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            k = 0
            for batch in train_loader:
                tracks = batch['track']
                true_sep = batch['separation']


                tracks = tracks.to(device=device, dtype=torch.float32)
                true_sep = true_sep.to(device=device, dtype=torch.float32)

                with torch.cuda.amp.autocast(enabled=True):
                    sep_preds = net(tracks)
                    loss = criterion(sep_preds, true_sep) # \
                           #+ dice_loss(F.softmax(masks_pred, dim=1).float(),
                           #            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                           #            multiclass=True)
                    loss /= float(sep_preds.size(2))
                    #print(masks_pred.size(2))
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    

                
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                epoch_loss += loss.item()
                #experiment.log({
                #    'train loss': loss.item(),
                #    'step': global_step,
                #    'epoch': epoch
                #})
                
                #if k % 100 == 0 :
                pbar.update(tracks.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})   
                    
                k += 1
                # Evaluation round
                division_step = (n_train // (4 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        
                        val_score = evaluate(net, val_loader, device)
                        
                        f_loss.write(str(loss.item()))
                        f_loss.write('\n')
                        f_l2.write(str(val_score))
                        f_l2.write('\n')
                        
                        #scheduler.step(val_score)

                        logging.info('Validation L2 norm: {}'.format(val_score))
                        #experiment.log({
                        #    'learning rate': optimizer.param_groups[0]['lr'],
                        #    'validation Dice': val_score,
                        #    'step': global_step,
                        #    'epoch': epoch,
                        #    **histograms
                        #})
            


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
            

if __name__ == '__main__':
    args = {'epochs' : 80,
            'batch_size' : 3,
            'lr' : 2e-5,
            'load' : True,
            'val' : 10.0,
            'classes' : 1,
            'bilinear': False
            }

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    net = UNet(n_channels=1, n_classes=args['classes'], bilinear=args['bilinear'])

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args['load']:
        #checkpoint = torch.load(PATH)
        #model.load_state_dict(checkpoint['model_state_dict'])
        net.load_state_dict(torch.load('./checkpoints/checkpoint_epoch80.pth', map_location=device))
        #logging.info(f'Model loaded from {args.load}')
        net.to(device=device)
        
        test_net(net, device)
        
    if False :
        net.to(device=device)
        try:
            train_net(net=net,
                      epochs=args['epochs'],
                      batch_size=args['batch_size'],
                      learning_rate=args['lr'],
                      device=device,
                      val_percent=args['val'] / 100)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            raise
