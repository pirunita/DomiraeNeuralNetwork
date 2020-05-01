import os
import torch


def make_directory(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def save_checkpoint(state, filename):
    torch.save(state, filename)