import argparse
import os
from os.path import isfile, join
import numpy as np

from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm

from hashes.dinohash import preprocess, dinohash
import torch
import torch.nn.functional as F
from apgd_attack import APGDAttack

def apgd_attack(image_files, n_iter=50):
    images = torch.stack([preprocess(Image.open(image_file)) for image_file in image_files])
    adv_images, loss = apgd.attack_single_run(images, n_iter)

    clean_PIL_images = [T.ToPILImage()(img) for img in images]
    adv_PIL_images = [T.ToPILImage()(img) for img in adv_images]

    for clean_img, adv_img, image_file in zip(clean_PIL_images, adv_PIL_images, image_files):
        name = image_file.split('/')[-1]
        clean_img.save(f'./adversarial_dataset/clean_{name}')
        adv_img.save(f'./adversarial_dataset/adv_{name}')

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./dataset', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=50,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)

image_files = [join(args.image_dir, f) for f in os.listdir(args.image_dir) if isfile(join(args.image_dir, f))]
batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]

apgd = APGDAttack(eps=args.epsilon)

for batch in tqdm(batches):
    apgd_attack(batch, n_iter=args.n_iter)