from robustbench.utils import load_model, clean_accuracy
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



parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="bpda", help='attack')
parser.add_argument('--input_defense', type=str, default="disco", help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--defense_disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=100, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--recursive_num', type=int, default=1, help='recursive_num for disco')
parser.add_argument('--adaptive', default=False, action="store_true", help='use adaptive attack')

args = parser.parse_args()
assert args.trial is not None

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



class AdaptiveAttackModel(nn.Module):
    def __init__(self, model_name, disco_path, device, adaptive):
        super().__init__()
        self.model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf').to(device)
        self.adaptive = adaptive
        if self.adaptive:
            self.defense = inr_adaptive_attack.INRAdaptiveAttack(device, disco_path, height=32, width=32)

    def forward(self, x):
        if self.adaptive:
            x = self.defense.forward(x)
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
        root = "log/bpda_results/debug"
    else:
        root = "log/bpda_results" 


    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(os.path.join(root, args.dataset)):
        os.mkdir(os.path.join(root, args.dataset))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense)):
        os.mkdir(os.path.join(root,  args.dataset, args.input_defense))
    if not os.path.exists(os.path.join(root,  args.dataset, args.input_defense, args.model_name)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name))
    if not os.path.exists(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack)):
        os.mkdir(os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack))


    filename = os.path.join(root, args.dataset, args.input_defense, args.model_name, args.attack, "result_" + args.trial + ".txt") 
 

    f = open(filename, "w")
    json.dump(args.__dict__, f, indent=2)
    f.write("\n")

    device = torch.device('cuda')
    batch_size=args.batch_size
    if args.debug:
        n_examples=50
    else:
        n_examples=args.batch_size
    
    model = AdaptiveAttackModel(model_name=args.model_name, disco_path=args.disco_path, device=device, adaptive=args.adaptive)

    classifier = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf').to(device)


    if args.attack == "bpda":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8/255,
            nb_iter=10, eps_iter=1/255, rand_init=False, clip_min=0.0, clip_max=1.0,
            targeted=False)
 

    defense = inr_adaptive_attack.INR(device, args.defense_disco_path, height=32, width=32)

    clean_acc_lst = []
    robust_acc_lst = []
    for i in range(args.repeat):
        print("==============")
        f.write("==============\n")
        print(str(i) + " Evaluation")
        f.write(str(i) + " Evaluation\n")


        clean_acc_all, robust_acc_all = 0, 0
        test_loader = load_cifar10(n_examples)
        iteration = int(10000/n_examples) if not args.debug else 1
        for idx, (x_clean, y_clean) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
 
            x_adv = adversary.perturb(x_clean, y_clean)

            for _ in range(args.recursive_num):
                x_clean = defense.forward(x_clean)
                x_adv = defense.forward(x_adv)


            clean_acc = clean_accuracy(classifier, x_clean, y_clean, batch_size=batch_size, device=device)
   
            clean_acc_all = clean_acc_all + clean_acc*n_examples/100.0
            robust_acc = clean_accuracy(classifier, x_adv, y_clean, batch_size=batch_size, device=device)
            robust_acc_all = robust_acc_all + robust_acc*n_examples/100.0

            if args.debug:
                break


        total = n_examples*iteration

        clean_acc_all = clean_acc_all/total*100.0
        print(f'Clean accuracy: {clean_acc_all:.4%}')
        f.write(f'Clean accuracy: {clean_acc_all:.4%}\n')
        clean_acc_lst.append(clean_acc_all)

        robust_acc_all = robust_acc_all/total*100.0
        print(f'Robust accuracy: {robust_acc_all:.4%}')
        f.write(f'Robust accuracy: {robust_acc_all:.4%}\n')
        robust_acc_lst.append(robust_acc_all)
        print("==============")
        f.write("==============\n")


    clean_acc_npy = np.array(clean_acc_lst)
    clean_acc_avg = np.mean(clean_acc_npy)
    clean_acc_std = np.std(clean_acc_npy)
     

    robust_acc_npy = np.array(robust_acc_lst)
    robust_acc_avg = np.mean(robust_acc_npy)
    robust_acc_std = np.std(robust_acc_npy)

    print(f'Avg clean accuracy: {clean_acc_avg:.4%}')
    f.write(f'Avg clean accuracy: {clean_acc_avg:.4%}\n')
    print(f'Std clean accuracy: {clean_acc_std:.4%}')
    f.write(f'Std clean accuracy: {clean_acc_std:.4%}\n')

    print(f'Avg robust accuracy: {robust_acc_avg:.4%}')
    f.write(f'Avg robust accuracy: {robust_acc_avg:.4%}\n')
    print(f'Std robust accuracy: {robust_acc_std:.4%}')
    f.write(f'Std robust accuracy: {robust_acc_std:.4%}\n')
 
    f.close()

if __name__ == "__main__":
    main()

