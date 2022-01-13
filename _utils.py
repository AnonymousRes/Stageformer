import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import logging
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset


def dp_predictor_loss(y_true, y_pred, mask_):
    ytr = y_true
    ypr = y_pred

    ymask = torch.unsqueeze(mask_, -1)
    masked_y = ypr * ymask

    batchsize_ = y_true.size()[0]
    loss = ytr * torch.log(masked_y + 1e-7) + (1 - ytr) * torch.log(1 - masked_y + 1e-7)
    loss = torch.sum(loss, dim=1) / (torch.sum(ymask, dim=1) + 1e-7)
    loss = torch.neg(torch.sum(loss)) / batchsize_
    return loss


def ihmp_predictor_loss(y_true, y_pred):
    ytr = y_true
    ypr = y_pred
    batchsize_ = y_true.size()[0]
    loss = ytr * torch.log(ypr + 1e-7) + (1 - ytr) * torch.log(1 - ypr + 1e-7)
    loss = torch.neg(torch.sum(loss)) / batchsize_
    return loss


class DPDataset(Dataset):
    def __init__(self, dpdata_file):
        self.dpdata_file = dpdata_file
        data_dict_file = open(self.dpdata_file, 'rb')
        data_dict = pickle.load(data_dict_file)
        data_dict_file.close()
        self.X = data_dict['X']
        self.MASK = data_dict['MASK']
        self.CUR_MASK = data_dict['CUR_MASK']
        self.INTERVAL = data_dict['INTERVAL']
        self.Y = data_dict['Y']


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.MASK[idx]
        curmask = self.CUR_MASK[idx]
        interval = self.INTERVAL[idx]
        y = self.Y[idx]

        return x, mask, curmask, interval, y


class IHMPDataset(Dataset):
    def __init__(self, dpdata_file):
        self.dpdata_file = dpdata_file
        data_dict_file = open(self.dpdata_file, 'rb')
        data_dict = pickle.load(data_dict_file)
        data_dict_file.close()
        self.X = data_dict['X']
        self.MASK = data_dict['MASK']
        self.CUR_MASK = data_dict['CUR_MASK']
        self.INTERVAL = data_dict['INTERVAL']
        self.Y = data_dict['Y']


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.MASK[idx]
        curmask = self.CUR_MASK[idx]
        interval = self.INTERVAL[idx]
        y = self.Y[idx]

        return x, mask, curmask, interval, y


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        # self.trace_func = trace_func
    def __call__(self, val_loss, n_epoch, emodel):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, n_epoch, emodel)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, n_epoch, emodel)
            self.counter = 0

    def save_checkpoint(self, val_loss, n_epoch, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logging.info(f'Epoch:{n_epoch+1:d} Validation AUPRC ({-1 * self.val_loss_min:.6f} --> {-1 * val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
