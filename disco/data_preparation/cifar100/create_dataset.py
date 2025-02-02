import numpy as np
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
import argparse
import timm

# attack
from pytorch_ares.attack_torch import FGSM
from pytorch_ares.attack_torch import PGD
from pytorch_ares.attack_torch import BIM

from wideresnet import WideResNet

parser = argparse.ArgumentParser()
parser.add_argument('--attack', help= 'attack method', default="pgd", choices= ['bim', 'pgd', 'fgsm'])
parser.add_argument('--debug', default=False, action="store_true")

args = parser.parse_args()

class cifar100_model(nn.Module):
    def __init__(self, model_name="WideResNet-28-10", layers=28, widen_factor=10, droprate=0, mean=[125.3/255.0, 123.0/255.0, 113.9/255.0], std=[63.0/255.0, 62.1/255.0, 66.7/255.0]):
        super(cifar100_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create 
        self.model = WideResNet(depth=layers, num_classes=100,  widen_factor=widen_factor, dropRate=droprate)

        checkpoint = torch.load("WideResNet_28_10_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x=self.transform(x)
        return self.model(x)



def imsave(img, title):
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(img)
    plt.imsave(title + ".png", img)


def main(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    net = cifar100_model().to(device)
    net.eval()

    dataset_name = 'cifar100'
    if args.attack == "fgsm": 
        norm = np.inf
        eps = 8/255.0
        target = False
        loss = 'ce'
        attack = FGSM(net, p=norm, eps=eps, data_name=dataset_name, target=target, loss=loss, device=device)

    elif args.attack == "pgd":
        target = False
        loss = 'ce'
        norm = np.inf
        eps = 8/255.0
        steps = 100
        stepsize = 2/255.0
        attack = PGD(net, epsilon=eps, norm=norm, stepsize=stepsize, steps=steps, data_name=dataset_name, target=target, loss=loss, device=device)

    elif args.attack == "bim":
        target = False
        loss = 'ce'
        p = np.inf
        eps = 8/255.0
        steps = 100
        stepsize = 2/255.0
        attack = BIM(net, epsilon=eps, p=p, stepsize=stepsize, steps=steps, data_name=dataset_name, target=target, loss=loss, device=device)
 
   
    name = attack.__class__.__name__

    save_dir = "dataset"
    atk_method = args.attack
   
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, atk_method)):
        os.mkdir(os.path.join(save_dir, atk_method))
    if not os.path.exists(os.path.join(save_dir, atk_method, "train")):
        os.mkdir(os.path.join(save_dir, atk_method, "train"))
        os.mkdir(os.path.join(save_dir, atk_method, "train/clean"))
        os.mkdir(os.path.join(save_dir, atk_method, "train/adv"))
    if not os.path.exists(os.path.join(save_dir, atk_method, "val")):
        os.mkdir(os.path.join(save_dir, atk_method, "val"))
        os.mkdir(os.path.join(save_dir, atk_method, "val/clean"))
        os.mkdir(os.path.join(save_dir, atk_method, "val/adv"))



    # creating training dataset
    cifar100_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'clean_data/CIFAR100')
    transform = transforms.Compose([transforms.ToTensor()])
    data_train = CIFAR100(root=cifar100_path, train=True, download=True, transform=transform)
    train_loader = DataLoader(data_train, batch_size=64, shuffle=False, num_workers=4, pin_memory= False, drop_last= False)
 
    cnt = 0
    for i, (img,labels) in tqdm(enumerate(train_loader, 1)):
        batchsize = img.shape[0]
        img, labels = img.to(device), labels.to(device)
            
        adv_images= attack.forward(img, labels, None)

        for i in range(img.shape[0]):
            image_np = np.transpose(img[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(image_np, os.path.join(save_dir, atk_method, "train/clean", str(cnt)))
        
            adv_image_np = np.transpose(adv_images[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(adv_image_np, os.path.join(save_dir, atk_method, "train/adv", str(cnt)))
            cnt += 1

            if args.debug:
                break
        if args.debug:
            break
         
    # creating testing dataset
    cifar100_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'clean_data/CIFAR100')
    transform = transforms.Compose([transforms.ToTensor()])
    data_test = CIFAR100(root=cifar100_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(data_test, batch_size=64, shuffle=False, num_workers=4, pin_memory= False, drop_last= False)
 
    cnt = 0
    for i, (img,labels) in tqdm(enumerate(test_loader, 1)):
        batchsize = img.shape[0]
        img, labels = img.to(device), labels.to(device)
            
        # Adv image without defense
        adv_images= attack.forward(img, labels, None)


        for i in range(img.shape[0]):
            image_np = np.transpose(img[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(image_np, os.path.join(save_dir, atk_method, "val/clean", str(cnt)))
        
            adv_image_np = np.transpose(adv_images[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(adv_image_np, os.path.join(save_dir, atk_method, "val/adv", str(cnt)))
            cnt += 1

            if args.debug:
                break
        if args.debug:
            break
 

if __name__ == "__main__":
    main(args)

