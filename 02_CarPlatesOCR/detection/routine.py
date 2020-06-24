import os
import sys
import time
import tqdm

from IPython.display import clear_output
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader
from utils import get_logger, dice_coeff, dice_loss



def eval_net(net, dataset, device):
    net.eval()
    tot = 0.
    with torch.no_grad():
        for i, b in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs, true_masks = b
            masks_pred = net(imgs.to(device)).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
            tot += dice_coeff(masks_pred.cpu(), true_masks).item()
    return tot / len(dataset)


def train(net, optimizer, criterion, scheduler, epochs, train_dataloader, val_dataloader, saveto, device, logger, show_plots=True):
    since = time.time()    
    
    num_batches = len(train_dataloader)
    best_model_info = {'epoch': -1, 'val_dice': 0., 'train_dice': 0., 'train_loss': 0.}
    
    bce_history=[]
    dice_history=[]
    loss_history=[]
    valDice_history=[]

    for epoch in range(epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        if scheduler is not None:
            scheduler.step(epoch)

        epoch_loss = 0.
        bce_epochHistory, dice_epochHistory = [], []
        tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, batch in tqdm_iter:
            imgs, true_masks = batch
            masks_pred = net(imgs.to(device))
            masks_probs = F.sigmoid(masks_pred)

            bce_val, dice_val = criterion(masks_probs.cpu().view(-1), true_masks.view(-1))
            loss = bce_val + dice_val
            
            bce_epochHistory.append(bce_val.item())
            dice_epochHistory.append(dice_val.item())
            epoch_loss += loss.item()
            
            tqdm_iter.set_description('mean loss: {:.4f}'.format(epoch_loss / (i + 1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if show_plots:
                if (i+1)%40==0:
                    clear_output(True)
                    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
                    ax[0][0].plot(bce_epochHistory, label='epoch bce loss')
                    ax[0][0].set_xlabel('batch')
                    ax[0][0].set_title('loss')
                    ax[0][1].plot(dice_epochHistory, label='epoch dice loss')
                    ax[0][1].set_xlabel('batch')
                    ax[0][1].set_title('loss')
                    ax[1][0].plot(bce_history, label='all bce loss')
                    ax[1][0].set_xlabel('epoch')
                    ax[1][0].set_title('loss')
                    ax[1][1].plot(dice_history, label='all dice loss')
                    ax[1][1].set_xlabel('epoch')
                    ax[1][1].set_title('loss')
                    ax[2][0].plot(loss_history, label='main loss (sum dice+bce)')
                    ax[2][0].set_xlabel('epoch')
                    ax[2][0].set_title('loss')
                    ax[2][1].plot(valDice_history, label='val dice')
                    ax[2][1].set_xlabel('epoch')
                    ax[2][1].set_title('val dice')
                    plt.legend()
                    plt.show()
                    

        logger.info('Epoch finished! Loss: {:.5f} ({:.5f} | {:.5f})'.format(epoch_loss / num_batches,
                                                                            np.mean(bce_epochHistory), np.mean(dice_epochHistory)))
        bce_history.append(np.mean(bce_epochHistory))
        dice_history.append(np.mean(dice_epochHistory))
        loss_history.append(epoch_loss / num_batches)

        val_dice = eval_net(net, val_dataloader, device=device)
        valDice_history.append(val_dice)
        
        if val_dice > best_model_info['val_dice']:
            best_model_info['val_dice'] = val_dice
            best_model_info['train_loss'] = epoch_loss / num_batches
            best_model_info['epoch'] = epoch
            torch.save(net.state_dict(), os.path.join(saveto, 'detectionbest.pth'))
            logger.info('Validation Dice Coeff: {:.5f} (best)'.format(val_dice))
        else:
            logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))

        torch.save(net.state_dict(), os.path.join(saveto, 'detectionlast.pth'))
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best epoch:{:4f} val_dice:{:4f} train_loss:{:4f}'.format(best_model_info['epoch'], best_model_info['val_dice'], best_model_info['train_loss']))
    