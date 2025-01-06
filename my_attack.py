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

def pgd_attack(images, num_iters=50):
    flipped_bits = ((outputs > 0.5).float() - original_hash).abs()

    clean_PIL_images = [T.ToPILImage()(img) for img in inputs]
    adv_PIL_images = [T.ToPILImage()(img) for img in x]

    for clean_img, adv_img, name in zip(clean_PIL_images, adv_PIL_images, images):
        name = name.split('/')[-1]
        clean_img.save(f'./dataset/clean_{name}')
        adv_img.save(f'./dataset/adv_{name}')

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./dataset', help='directory containing images')
parser.add_argument('--alpha', dest='alpha', type=float, default=2/255,
                    help='step size for each iteration')
parser.add_argument('--num_iter', dest='num_iter', type=int, default=50,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')

args = parser.parse_args()

image_files = [join(args.image_dir, f) for f in os.listdir(args.image_dir) if isfile(join(args.image_dir, f))]

batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]

for batch in tqdm(batches):
    pgd_attack(batch, alpha=args.alpha, num_iter=args.num_iter, epsilon=args.epsilon)