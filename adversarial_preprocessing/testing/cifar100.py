from robustbench.utils import load_model, clean_accuracy
from autoattack import AutoAttack
import torch
import os
from robustbench.model_zoo.defense import bit_depth_reduction
from robustbench.model_zoo.defense import jpeg_compression 
from robustbench.model_zoo.defense import randomization
from robustbench.model_zoo.defense import inr
from robustbench.model_zoo.defense import autoencoder
from robustbench.model_zoo.defense import stl
from torchvision import transforms
import argparse
import json
import torch
import torch.nn as nn
from wideresnet import WideResNet
import numpy as np
import torch.nn as nn
from advertorch.defenses import MedianSmoothing2D
from tqdm import tqdm
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="autoattack", help='attack')
parser.add_argument('--input_defense', type=str, help='defense type')
parser.add_argument('--model_name', type=str, default="WideResNet-34-10", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--stl_npy_name', type=str, default="64_p8_lm0.1", help='path to the stl numpy file')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=128, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="cifar100", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')

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




class cifar100_model(nn.Module):
    def __init__(self, model_name, layers, widen_factor, droprate=0, mean=[125.3/255.0, 123.0/255.0, 113.9/255.0], std=[63.0/255.0, 62.1/255.0, 66.7/255.0]):
        super(cifar100_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create model
        self.model = WideResNet(depth=layers, num_classes=100,  widen_factor=widen_factor, dropRate=droprate)

        #model = model.cuda()
        checkpoint = torch.load("WideResNet-pytorch/runs/" + model_name + "/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        x=self.transform(x)
        return self.model(x)

def load_cifar100(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    
    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=4)
    
    return test_loader
 
def main():
    print(args.input_defense)

    if args.debug:
        root = "log/results/debug"
    else:
        root = "log/results"
     
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
    n_examples=args.batch_size
    

    if args.model_name == "WideResNet-28-10":
        model = cifar100_model(args.model_name, layers=28, widen_factor=10).to(device)
    elif args.model_name == "WideResNet-70-16":
        model = cifar100_model(args.model_name, layers=70, widen_factor=16).to(device)
    elif args.model_name == "WideResNet-34-10":
        model = cifar100_model(args.model_name, layers=34, widen_factor=10).to(device)
    else:
        model = load_model(model_name=args.model_name, dataset='cifar100', threat_model='Linf').to('cuda')
        
 
    if args.debug:
        if args.norm == "Linf":
            adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
            adversary.apgd.n_restarts = 1
        elif args.norm == "L2":
            adversary = AutoAttack(model, norm='L2', eps=0.5, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
            adversary.apgd.n_restarts = 1
    else:
        if args.norm == "Linf":
            if args.model_name == "WideResNet-70-16":
                print("WideResNet-70-16")
                adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                adversary.apgd.n_restarts = 1
            else:
                adversary = AutoAttack(model, norm='Linf', eps=8/255)
        elif args.norm == "L2":
            adversary = AutoAttack(model, norm='L2', eps=0.5)


    if args.input_defense == "disco":
        defense = inr.INR(device, args.disco_path, height=32, width=32)
    elif args.input_defense == "bit_depth_reduction":
        defense = bit_depth_reduction.BitDepthReduction(device)
    elif args.input_defense == "jpeg":
        defense = jpeg_compression.Jpeg_compresssion(device)
    elif args.input_defense == "randomization":
        defense = randomization.Randomization(device)
    elif args.input_defense == "stl":
        defense = stl.STL(device, "cifar", args.stl_npy_name)
    elif args.input_defense == "medianfilter":
        defense = nn.Sequential(MedianSmoothing2D(kernel_size=3))

    clean_acc_lst = []
    robust_acc_lst = []
    for i in range(args.repeat):
        print("==============")
        f.write("==============\n")
        print(str(i) + " Evaluation")
        f.write(str(i) + " Evaluation\n")


        clean_acc_all, robust_acc_all = 0, 0
        test_loader = load_cifar100(n_examples)
        iteration = int(10000/n_examples) if not args.debug else 1
        for idx, (x_clean, y_clean) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
 
            x_adv = adversary.run_standard_evaluation(x_clean, y_clean)

            if args.input_defense != "no_input_defense":
                x_clean = defense.forward(x_clean)
                x_adv = defense.forward(x_adv)

            clean_acc = clean_accuracy(model, x_clean, y_clean, batch_size=batch_size, device=device)
   
            clean_acc_all = clean_acc_all + clean_acc*n_examples/100.0
            robust_acc = clean_accuracy(model, x_adv, y_clean, batch_size=batch_size, device=device)
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

