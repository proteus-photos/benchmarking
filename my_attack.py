import argparse
import os
from os.path import isfile, join
import numpy as np

from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from hashes.dinohash import preprocess, dinohash, dinov2
from apgd_attack import APGDAttack

class ImageDataset(Dataset):
    def __init__(self, image_files):
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(args.image_dir, self.image_files[idx])).convert("RGB")
        return preprocess(image), self.image_files[idx]

def apgd_attack(image_tensors, names, n_iter):
    logits = dinohash(image_tensors, differentiable=False, logits=True, l2_normalize=False).float()
    adv_images, _ = apgd.attack_single_run(image_tensors, logits, n_iter)

    # clean_PIL_images = [T.ToPILImage()(img) for img in image_tensors]
    adv_PIL_images = [T.ToPILImage()(img) for img in adv_images]

    for adv_img, name in zip(adv_PIL_images, names):
        name = name.replace('.jpg', '.png')
        # clean_img.save(f'./adversarial_dataset/clean/{name}')
        adv_img.save(f'./adversarial_dataset/adv/{name}')

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./diffusion_data', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=30,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)
os.makedirs('./adversarial_dataset/adv', exist_ok=True)

image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
image_files.sort()
image_files = image_files[689_600:1_250_000]

dataset = ImageDataset(image_files)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

apgd = APGDAttack(eps=args.epsilon)
dinov2.eval()
for param in dinov2.parameters():
    param.requires_grad = False

for image_tensors, names in tqdm(dataloader):
    apgd_attack(image_tensors, names, n_iter=args.n_iter)