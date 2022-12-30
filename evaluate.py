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
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            #mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #mask_numpy = mask_true.cpu().numpy()
            #pred_numpy = mask_pred.cpu().numpy()
            
            
            l2_norm += np.linalg.norm(mask_true.cpu() - mask_pred.cpu())

           

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
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            loss /= float(mask_pred.size(2))
            mse_loss += loss.item()

            l2_norm += np.linalg.norm(mask_true.cpu() - mask_pred.cpu())
            

        
    #net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return l2_norm, mse_loss
    return (l2_norm / num_val_batches), (mse_loss / num_val_batches)
