import argparse
import logging
import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import tqdm

from tensorboardX import SummaryWriter
from utils.net import VGG16
from utils.util import make_directory

# Set logger
logger = logging.getLogger('DataLoader')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# Setting
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def get_args():
    parser = argparse.ArgumentParser(description='vgg')
    
    # Environment Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    
    # Parameter
    parser.add_argument('--name', type=str, default='VGG16', help='Trained model name')
    parser.add_argument('--mode', type=str, default='train', help='train / test')
    parser.add_argument('--session', type=int, default=1)
    parser.add_argument('--augmentation', type=bool, default=True, help='Data Augmentations')
    
    # Hyperparameter
    parser.add_argument('--lr', type=float, default=0.1, help='Initial Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epoch', dest='max_epoch', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.9)
    
    # Directory
    parser.add_argument('--dataroot', default='datasets')
    
    parser.add_argument('--log_path', default='logs')
    parser.add_argument('--checkpoints_path', default='checkpoints')
    parser.add_argument('--tensorboard_path', default='tensorboard')
    
    args = parser.parse_args()
    
    return args


def train(args, model, criterion, train_dataset, board, log_writer, checkpoint_dir, use_cuda):
    torch.manual_seed(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(5)
    
    if use_cuda:
        model.cuda()
        model.train()
        criterion.cuda()
    else:
        model.train()
    
    lr = args.lr
    
    
    
    # Setting Optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
                
    
    optimizer = torch.optim.SGD(params, momentum=args.momentum)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)    
    
    for epoch in tqdm.tqdm(range(args.start_epoch, args.max_epoch + 1), desc='Training'):
        loss_sum = 0
    
        for i, (img, target) in enumerate(train_loader):
            if use_cuda:
                img = input.cuda()
                target = target.cuda()
            
            output = model(img)
            loss = criterion(output, target)
            
            # Parameter update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()



if __name__=='__main__':
    args = get_args()
    logger.info(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    # Model
    model = VGG16(args)
    
    if args.mode == 'train':
        root_dir = args.train
        session_dir = os.path.join(root_dir, str(args.session))
        
        log_dir = os.path.join(session_dir, args.log_path)
        checkpoints_dir = os.path.join(session_dir, args.checkpoints_path)
        tensorboard_dir = os.path.join(session_dir, args.tensorboard_path)
        
        make_directory(root_dir, session_dir, log_dir, checkpoints_dir, tensorboard_dir)
        
        # Dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]), download=True)
        
        # Define loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Trained model Summary Writer
        board = SummaryWriter(logdir=os.path.join(tensorboard_dir))
        
        # Log Training record
        log_file_dir = os.path.join(log_dir, 'log_{}_{}.txt'.format(args.name, args.session))
        log_writer = open(log_file_dir, 'a')
        log_writer.write(str(args))
        log_writer.write('\n')
        
        # Train
        train(args, model, criterion, train_dataset, board, log_writer, checkpoints_dir)
    

