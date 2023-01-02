import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    l2_norm = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        signal, sep_true = batch['track'], batch['separation']
        # move images and labels to correct device and type
        signal = signal.to(device=device, dtype=torch.float32)
        sep_true = sep_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            sep_pred = net(signal)
            
            l2_norm += np.linalg.norm(sep_true.cpu() - sep_pred.cpu())

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return l2_norm
    return l2_norm / num_val_batches



def evaluate_with_loss(net, dataloader, device, criterion):
    #net.eval()
    num_val_batches = len(dataloader)
    l2_norm = 0
    mse_loss = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        signal, sep_true = batch['track'], batch['separation']
        # move images and labels to correct device and type
        signal = signal.to(device=device, dtype=torch.float32)
        sep_true = sep_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            sep_pred = net(signal)
            loss = criterion(sep_pred, sep_true)
            loss /= float(sep_pred.size(2))
            mse_loss += loss.item()

            l2_norm += np.linalg.norm(sep_true.cpu() - sep_pred.cpu())
            

        
    #net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return l2_norm, mse_loss
    return (l2_norm / num_val_batches), (mse_loss / num_val_batches)
