import os
import torch
import torch.utils.data as data

class CIFAR10(data.Dataset):
    def __init__(self, args):
        self.data_path = args.data_path
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_list = [
            ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
            ['data_batch_2', 'd4bba439e000b95fd0a9b']
        ]
def make_directory(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

def save_checkpoint(state, filename):
    torch.save(state, filename)
