from robustbench.utils import load_model, clean_accuracy
from autoattack import AutoAttack
import torch
import torch.nn as nn
import os
from robustbench.model_zoo.defense import inr_adaptive_attack
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse
import json
import numpy as np
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from advertorch.bpda import BPDAWrapper
from cifar10_models.vgg import vgg16_bn
from torchvision import transforms
import time



parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="autoattack", help='attack')
parser.add_argument('--input_defense', type=str, default="disco", help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--defense_disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--batch_size', type=int, default=1, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--recursive_num', type=int, default=1, help='recursive_num for disco')
parser.add_argument('--adaptive', default=False, action="store_true", help='use adaptive attack')
parser.add_argument('--adaptive_iter', type=int, default=1, help='how many adaptive')
parser.add_argument('--measure', type=str, help='measure')

args = parser.parse_args()

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}

class cifar10_model(nn.Module):
    def __init__(self, model, mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]):
        super(cifar10_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model = model
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        
    def forward(self, x):
        x=self.transform(x)
        return self.model(x)



class AdaptiveAttackModel(nn.Module):
    def __init__(self, model_name, disco_path, device, adaptive, adaptive_iter):
        super().__init__()
        if args.model_name == "vgg16_bn":
            self.model = cifar10_model(vgg16_bn(pretrained=True)).to('cuda')
        else:
            self.model = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf').to('cuda')
 

        self.adaptive = adaptive
        if self.adaptive: 
            self.defense = []
            self.adaptive_iter = adaptive_iter
            print("========")
            print(self.adaptive_iter)
            print("========")
            for _ in range(self.adaptive_iter):
                self.defense.append(inr_adaptive_attack.INRAdaptiveAttack(device, disco_path, height=32, width=32))

    def forward(self, x):
        if self.adaptive:
            for i in range(self.adaptive_iter):
                x = self.defense[i].forward(x)
        return self.model(x)

def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    
    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=4)
    
    return test_loader
 


def main():

    if args.debug:
        root = "log/bpda_speed/debug"
    else:
        root = "log/bpda_speed" 


    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, args.measure)):
        os.mkdir(os.path.join(root, args.measure))


    if args.measure == "attack":
        filename = os.path.join(root, args.measure, "result_" + str(args.adaptive_iter) + ".txt") 
    elif args.measure == "defense":
        filename = os.path.join(root, args.measure, "result_" + str(args.recursive_num) + ".txt") 

    f = open(filename, "w")
    json.dump(args.__dict__, f, indent=2)
    f.write("\n")

    device = torch.device('cuda')
    batch_size=args.batch_size
    n_examples=args.batch_size
    
    model = AdaptiveAttackModel(model_name=args.model_name, disco_path=args.disco_path, device=device, adaptive=args.adaptive, adaptive_iter=args.adaptive_iter)

    classifier = cifar10_model(vgg16_bn(pretrained=True)).to('cuda')

    adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
            nb_iter=10, eps_iter=1/255, rand_init=False, clip_min=0.0, clip_max=1.0,
            targeted=False)

    defense = inr_adaptive_attack.INR(device, args.disco_path, height=32, width=32)


    fps_lst, time_lst = [], []
    for i in range(args.repeat):

        test_loader = load_cifar10(n_examples)
        elapse_time = 0
        for idx, (x_clean, y_clean) in enumerate(test_loader):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
 
            if args.measure == "attack":
                start = time.time()
                x_adv = adversary.perturb(x_clean, y_clean)
                diff = time.time() - start
                elapse_time += diff
            elif args.measure == "defense":
                start = time.time()
                for _ in range(args.recursive_num):
                    x_clean = defense.forward(x_clean)
                diff = time.time() - start
                elapse_time += diff
           
            if idx == 100:
                break
 
        f.write("=======\n")
        fps = 100.0/elapse_time
        print("Attack Time :" + str(elapse_time) + " FPS: " + str(fps))
        f.write("Attack Time :" + str(elapse_time) + " FPS: " + str(fps) + "\n")
        time_lst.append(elapse_time)
        fps_lst.append(fps)
        f.write("=======\n")
 
  
    time_mean = np.mean(np.asarray(time_lst)) 
    time_std = np.std(np.asarray(time_lst)) 
    print("Time mean:" + str(time_mean) + " Time std:" + str(time_std))
    f.write("Time mean:" + str(time_mean) + " Time std:" + str(time_std) + "\n")

    fps_mean = np.mean(np.asarray(fps_lst)) 
    fps_std = np.std(np.asarray(fps_lst)) 
    print("Fps mean:" + str(fps_mean) + " fps std:" + str(fps_std))
    f.write("Fps mean:" + str(fps_mean) + " fps std:" + str(fps_std) + "\n")
 
    f.close()

if __name__ == "__main__":
    main()

