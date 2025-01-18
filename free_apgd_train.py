import argparse
import os

from PIL import Image
import torch
from torch.optim import AdamW
from tqdm import tqdm

import numpy as np
from hashes.dinohash import preprocess, dinov2, dinohash
import torch
from apgd_attack import APGDAttack, criterion_loss

torch.manual_seed(0)
np.random.seed(0)

def adjust_learning_rate(initial_lr, optimizer, epoch, n_repeats):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // int(math.ceil(30./n_repeats))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.hashes = []
        batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]
        for batch in tqdm(batches):
            hashes = dinohash([Image.open(image_file) for image_file in batch], differentiable=False).cpu()
            self.hashes.append(hashes)
        self.hashes = torch.cat(self.hashes).float()
        np.save('./hashes.npy', self.hashes.numpy())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = preprocess(Image.open(image_file))
        return image, self.hashes[idx]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./dataset', help='directory containing images')
parser.add_argument('--n_iter_min', dest='n_iter_min', type=int, default=5,
                    help='minimum number of iterations')
parser.add_argument('--n_iter_max', dest='n_iter_max', type=int, default=15,
                    help='maximum number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=2,
                    help='number of epochs')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)

image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))][:1024]

dataset = ImageDataset(image_files)
complete_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

apgd = APGDAttack(eps=args.epsilon)
optimizer = AdamW(dinov2.parameters(), lr=1e-6, weight_decay=2e-4)

# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import numpy as np
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--output_prefix', default='free_adv', type=str,
                    help='prefix used to define output path')
    parser.add_argument('-c', '--config', default='configs.yml', type=str, metavar='Path',
                    help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
    return parser.parse_args()


# Parase config file and initiate logging
cudnn.benchmark = True

def main():
    # Scale and initialize the parameters
    best_prec1 = 0
    configs.TRAIN.epochs = int(math.ceil(configs.TRAIN.epochs / configs.ADV.n_repeats))
    configs.ADV.fgsm_step /= configs.DATA.max_color_value
    configs.ADV.clip_eps /= configs.DATA.max_color_value
    
    # Wrap the model into DataParallel
    model = torch.nn.DataParallel(model).cuda()
    
    # Criterion:
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer:
    optimizer = torch.optim.SGD(model.parameters(), configs.TRAIN.lr,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)
                
    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(configs.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
        num_workers=configs.DATA.workers, pin_memory=True, sampler=None)
    
    normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                    std=configs.TRAIN.std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(configs.DATA.img_size),
            transforms.CenterCrop(configs.DATA.crop_size),
            transforms.ToTensor(),
        ])),
        batch_size=configs.DATA.batch_size, shuffle=False,
        num_workers=configs.DATA.workers, pin_memory=True)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        for pgd_param in configs.ADV.pgd_attack:
            validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)
        validate(val_loader, model, criterion, configs, logger)
        return
    
    
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        adjust_learning_rate(configs.TRAIN.lr, optimizer, epoch, configs.ADV.n_repeats)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, configs, logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': configs.TRAIN.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, os.path.join('trained_models', configs.output_name))
        
    # Automatically perform PGD Attacks at the end of training
    logger.info(pad_str(' Performing PGD Attacks '))
    for pgd_param in configs.ADV.pgd_attack:
        validate_pgd(val_loader, model, criterion, pgd_param[0], pgd_param[1], configs, logger)

        
# Free Adversarial Training Module        
global global_noise_data
global_noise_data = torch.zeros([configs.DATA.batch_size, 3, configs.DATA.crop_size, configs.DATA.crop_size]).cuda()
def train(train_loader, model, criterion, optimizer, epoch):
    global global_noise_data
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    # Initialize the meters

    # switch to train mode
    model.train()
    for i, (input, target) in enumerate(train_loader):
        end = time.time()
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        for j in range(configs.ADV.n_repeats):
            # Ascend on the global noise
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).cuda()
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)
            
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            
            # Update the noise for the next iteration
            pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-configs.ADV.clip_eps, configs.ADV.clip_eps)

            optimizer.step()
            # measure elapsed time

for epoch in range(args.n_epochs):

    for image_batch, hash_batch in tqdm(train_loader):
        n_iter = np.random.randint(args.n_iter_min, args.n_iter_max)

        hash_batch = hash_batch.cuda()
        adv_images, _ = apgd.attack_single_run(image_batch, hash_batch, n_iter)
        # adv_images = image_batch.cuda()

        optimizer.zero_grad()

        dinov2.train()

        adv_hash_batch, loss = criterion_loss(adv_images, hash_batch, loss="bce", l2_normalize=False)
        loss = loss.mean()

        accuracy = (adv_hash_batch - hash_batch).abs().mean()
        # print("Attack strength:", accuracy.item())

        loss.backward()
        optimizer.step()
        del hash_batch

    dinov2.eval()

    total_strength = 0
    n_images = 0

    for image_batch, hash_batch in tqdm(test_loader):
        hash_batch = hash_batch.cuda()

        adv_images, _ = apgd.attack_single_run(image_batch, hash_batch, args.n_iter_max)

        adv_hash_batch = dinohash(adv_images, differentiable=False, tensor=True).float()

        accuracy = (adv_hash_batch - hash_batch).cpu().abs().mean().item()

        total_strength += accuracy * len(image_batch)
        n_images += len(image_batch)

        del hash_batch
    
    print("Validation attack strength:", total_strength / n_images)