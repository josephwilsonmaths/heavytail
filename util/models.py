import torch
import tqdm
import math

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.normal_(m.bias,mean=0,std=1)

"""Fully Connected ResNet architecture."""

from collections.abc import Callable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def get_optim_sched(network, optim, sched, lr, wd, T_max):
    if optim == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=wd)  
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=wd)    
    else:
        print("Invalid optimizer choice. Valid choices: [adam, sgd]")

    if sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max)
    elif sched == 'poly':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=T_max*10, power=0.5)
    elif sched == 'linear':
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=T_max, power=1)
    else:
        scheduler = None

    return optimizer, scheduler

def train(network, dataloader, optimizer, loss_fn, epochs, int_tol, scheduler=None, verbose=False):
    network.train()

    device = next(network.parameters()).device

    if verbose:
        pbar = tqdm.trange(epochs)
    else:
        pbar = range(epochs)

    for epoch in pbar:

        epoch_loss = 0

        for x,y in dataloader:
            x,y = x.to(device), y.reshape(-1,1).to(device)
            prediction = network(x)
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)

        if epoch_loss <= int_tol:
            return epoch_loss
        
        if math.isnan(epoch_loss):
            return epoch_loss

        if verbose:
            pbar.set_description(f"MSE Loss: {epoch_loss:.4f}")

    return epoch_loss

def test(network, dataloader, loss_fn, verbose=False):
    network.eval()

    device = next(network.parameters()).device

    if verbose:
        pbar = tqdm.tqdm(dataloader)
    else:
        pbar = dataloader

    test_loss = 0

    for x,y in pbar:
        x,y = x.to(device), y.reshape(-1,1).to(device)

        prediction = network(x)
        test_loss += loss_fn(prediction, y).item()

    test_loss = test_loss / len(dataloader)
    return test_loss