import numpy as np
import torch
from torch import nn
import subprocess as sp
import os, math


class EarlyStoppingTorch:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path=None, dataset=None, missing_rate=0, seed=2024, patience=7, verbose=False, delta=0.0001):
        """
        Args:
            save_path : 
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.dataset = dataset
        self.missing_rate = missing_rate
        self.seed = seed
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.save_path:
            path = os.path.join(self.save_path, f'{self.dataset}_{str(self.missing_rate)}_{str(self.seed)}_best_network.pt')
            torch.save(model.state_dict(), path)	
        self.val_loss_min = val_loss
