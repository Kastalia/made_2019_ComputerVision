import os
import sys
import tqdm
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

import numpy as np
import torch, torch.nn as nn
from torch import optim
from torch.nn.functional import ctc_loss, log_softmax
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from recognition.transform import Compose, Resize, Pad, Rotate
from recognition.model import RecognitionModel
from recognition.dataset import RecognitionDataset
from recognition.common import abc
import editdistance
#
from utils import get_logger


def eval(net, data_loader, device):
    count, tp, avg_ed = 0, 0, 0
    iterator = tqdm.tqdm(data_loader)
    
    with torch.no_grad():
        for batch in iterator:
            images = batch['images'].to(device)
            out = net(images, decode=True)
            gt = (batch['seqs'].numpy() - 1).tolist()
            lens = batch['seq_lens'].numpy().tolist()
            
            pos, key = 0, ''
            for i in range(len(out)):
                gts = ''.join(abc[c] for c in gt[pos:pos + lens[i]])
                pos += lens[i]
                if gts == out[i]:
                    tp += 1
                else:
                    avg_ed += editdistance.eval(out[i], gts)
                count += 1
    
    acc = tp / count
    avg_ed = avg_ed / count
    
    return acc, avg_ed

  
def train(net, optimizer, criterion, scheduler, epochs, train_dataloader, val_dataloader, saveto, device, logger, show_plots=True):
    since = time.time() 
    # TODO: try different techniques for fighting overfitting of the trained network
    best_acc_val = -1
    for e in range(epochs):
        logger.info('Starting epoch {}/{}.'.format(e + 1, epochs))
        
        net.train()
        if scheduler is not None:
            scheduler.step()
            
        loss_mean = []
        train_iter = tqdm.tqdm(train_dataloader)
        for j, batch in enumerate(train_iter):
            optimizer.zero_grad()
            images = batch['images'].to(device)
            seqs = batch['seqs']
            seq_lens = batch['seq_lens']
            
            seqs_pred = net(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()
            # TODO: ctc_loss is not an only choice here
            loss = criterion(log_probs, seqs, seq_lens_pred, seq_lens) #/ batch_size
            loss.backward()
            loss_mean.append(loss.item())
            
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            
        logger.info('Epoch finished! Loss: {:.5f}'.format(np.mean(loss_mean)))

        net.eval()
        acc_val, acc_ed_val = eval(net, val_dataloader, device=device)

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            torch.save(net.state_dict(), os.path.join(saveto, 'recognition_best.pth'))
            logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best)'.format(acc_val, acc_ed_val))
        else:
            logger.info('Valid acc: {:.5f}, acc_ed: {:.5f} (best {:.5f})'.format(acc_val, acc_ed_val, best_acc_val))

        torch.save(net.state_dict(), os.path.join(saveto, 'recognition_last.pth'))
    logger.info('Best valid acc: {:.5f}'.format(best_acc_val))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


