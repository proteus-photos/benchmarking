#from robustbench.data import load_imagenet
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
from robustbench.loaders import CustomImageFolder
import torch.utils.data as data
import torchvision.transforms as transforms
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import argparse
import json
import torchvision.models as models
import torch.nn as nn 
import numpy as np
from tqdm import tqdm
from advertorch.defenses import MedianSmoothing2D

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, default="autoattack", help='attack')
parser.add_argument('--input_defense', type=str, help='defense type')
parser.add_argument('--model_name', type=str, default="Standard_R50", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--stl_npy_name', type=str, default="64_p8_lm0.1", help='path to the stl numpy file')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=100, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="imagenet", help='dataset')
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
    'Res299':
    transforms.Compose([
        transforms.Resize((299,299)),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}

class imagenet_model(nn.Module):
    def __init__(self, model_name, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(imagenet_model, self).__init__()
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.transform = transforms.Compose([transforms.Normalize(self.mean, self.std)])
        # create model
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model_name == "inceptionv3":
            self.model = models.inception_v3(aux_logits=False, pretrained=True)
        elif model_name == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(pretrained=True)


    def forward(self, x):
        x=self.transform(x)
        return self.model(x)



def load_imagenet(
    n_examples: Optional[int] = 5000,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS['Res256Crop224']
) -> Tuple[torch.Tensor, torch.Tensor]:
    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)

    test_loader = data.DataLoader(imagenet,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=4)

    #x_test, y_test, paths = next(iter(test_loader))

    #return x_test, y_test
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
    if args.debug:
        n_examples=100
    else:
        n_examples=args.batch_size

    if args.input_defense == "disco":
        if args.model_name == "inceptionv3": 
            defense = inr.INR(device, args.disco_path, height=299, width=299)
        else:
            defense = inr.INR(device, args.disco_path, height=256, width=256)
    elif args.input_defense == "bit_depth_reduction":
        defense = bit_depth_reduction.BitDepthReduction(device)
    elif args.input_defense == "jpeg":
        defense = jpeg_compression.Jpeg_compresssion(device)
    elif args.input_defense == "randomization":
        defense = randomization.Randomization(device)
    elif args.input_defense == "stl":
        defense = stl.STL(device, "imagenet", args.stl_npy_name)
    elif args.input_defense == "medianfilter":
        defense = nn.Sequential(MedianSmoothing2D(kernel_size=3))



    clean_acc_lst = []
    robust_acc_lst = []

    for i in range(args.repeat): 
        print("==============")
        f.write("==============\n")
        print(str(i) + " Evaluation")
        f.write(str(i) + " Evaluation\n")

        if args.input_defense != "no_input_defense":
            if args.model_name == "Standard_R50":
                model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')
            elif args.model_name == "wide_resnet50_2" or args.model_name == "resnet18" or args.model_name == "inceptionv3":
                model = imagenet_model(args.model_name).to('cuda')     
            else:
                model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')
        else:
            if args.model_name == "wide_resnet50_2" or args.model_name == "resnet18" or args.model_name == "inceptionv3":
                model = imagenet_model(args.model_name).to('cuda')     
            else:
                model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model='Linf').to('cuda')

        model.eval() 
        iteration = int(5000/n_examples) if not args.debug else 1
        clean_acc_all, robust_acc_all = 0, 0
        if args.model_name == "inceptionv3":
            test_loader = load_imagenet(n_examples, transforms_test=PREPROCESSINGS['Res299'])
        else:
            test_loader = load_imagenet(n_examples)
        for idx, (x_clean, y_clean, path) in tqdm(enumerate(test_loader)):
            x_clean, y_clean = x_clean.to(device), y_clean.to(device)
            if args.debug:
                if args.norm == "Linf":
                    adversary = AutoAttack(model, norm='Linf', eps=4/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
                elif args.norm == "L2":
                    adversary = AutoAttack(model, norm='L2', eps=0.5, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                    adversary.apgd.n_restarts = 1
            else:
                if args.norm == "Linf":
                    adversary = AutoAttack(model, norm='Linf', eps=4/255)
                elif args.norm == "L2":
                    adversary = AutoAttack(model, norm='L2', eps=0.5)

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

