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
from torchvision.datasets import CIFAR10
from third_party.example.cifar10.pytorch_cifar10.models import *
import argparse
import timm

# attack
from pytorch_ares.attack_torch import FGSM
from pytorch_ares.attack_torch import PGD
from pytorch_ares.attack_torch import BIM


parser = argparse.ArgumentParser()
parser.add_argument('--attack', help= 'attack method', default="pgd", choices= ['bim', 'fgsm', 'pgd'])
parser.add_argument('--debug', default=False, action="store_true")
args = parser.parse_args()

class cifar10_model(torch.nn.Module):
    def __init__(self,device,model):
        torch.nn.Module.__init__(self)
        self.device=device
        self.model = model
        self.model = self.model.to(self.device)

    def forward(self, x):
        self.mean_torch_c = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1).to(self.device)
        self.std_torch_c = torch.tensor((0.2023, 0.1994, 0.2010)).view(3,1,1).to(self.device)
        x = (x - self.mean_torch_c) / self.std_torch_c
        labels = self.model(x.to(self.device))
        return labels

def imsave(img, title):
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(img)
    plt.imsave(title + ".png", img)


def main(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join('resnet18_ckpt.pth')
    model = ResNet18()
    pretrain_dict = torch.load(path, map_location=device)
    model.load_state_dict(pretrain_dict['net'])
    net = cifar10_model(device, model)
    net.eval()

    dataset_name = 'cifar10'
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
    cifar10_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'clean_data/CIFAR10')
    transform = transforms.Compose([transforms.ToTensor()])
    data_train = CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
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


    # creating testing dataset
    cifar10_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'clean_data/CIFAR10')
    transform = transforms.Compose([transforms.ToTensor()])
    data_test = CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(data_test, batch_size=64, shuffle=False, num_workers=4, pin_memory= False, drop_last= False)
 
    cnt = 0
    for i, (img,labels) in tqdm(enumerate(test_loader, 1)):
        batchsize = img.shape[0]
        img, labels = img.to(device), labels.to(device)
            
        adv_images= attack.forward(img, labels, None)

        for i in range(img.shape[0]):
            image_np = np.transpose(img[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(image_np, os.path.join(save_dir, atk_method, "val/clean", str(cnt)))
        
            adv_image_np = np.transpose(adv_images[i].cpu().detach().numpy(), (1, 2, 0))
            imsave(adv_image_np, os.path.join(save_dir, atk_method, "val/adv", str(cnt)))
            cnt += 1

        if args.debug:
            break

if __name__ == "__main__":
    main(args)

