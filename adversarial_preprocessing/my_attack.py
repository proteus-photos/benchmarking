import argparse
import os
from os.path import isfile, join
from random import randint
import traceback

from PIL import Image
import torch
import torchvision.transforms as T
from onnx import load_model
from skimage import feature
from skimage.color import rgb2gray
from tqdm import tqdm

from models.neuralhash import NeuralHash
from utils.hashing import compute_hash, load_hash_matrix
from utils.image_processing import save_images
from utils.logger import Logger
from metrics.hamming_distance import hamming_distance
import threading
import concurrent.futures
from itertools import repeat
import copy
import time
from dinohash import DinoExtractor, preprocess, dinohash, normalize
import torch
import torch.nn.functional as F


def pgd_attack(images, alpha=1/255, num_iter=50, epsilon=8/255):
    """
    Perform a Projected Gradient Descent (PGD) attack.

    Args:
        model: PyTorch model to attack.
        inputs: Original inputs (batch of images).
        labels: True labels for untargeted attack, or target labels for targeted attack.
        epsilon: Maximum perturbation (L∞ norm bound).
        alpha: Step size for each iteration.
        num_iter: Number of iterations.

    Returns:
        Adversarial examples.
    """

    inputs = torch.stack([preprocess(Image.open(img).convert('RGB')).cuda() for img in images])
    original_hash = (dinohash(inputs, differentiable=True) > 0.5).float()
    adv_inputs = inputs.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        # Forward pass
        outputs = dinohash(adv_inputs, differentiable=True, c=10)

        loss = torch.nn.functional.mse_loss(outputs, original_hash, reduction="mean")
        print(loss.item())
        loss.backward()

        with torch.no_grad():
            adv_inputs += alpha * adv_inputs.grad.sign()
            
            # project perturbations to L∞ norm ball
            adv_inputs.copy_(adv_inputs.clamp(inputs - epsilon, inputs + epsilon))
            adv_inputs.copy_(adv_inputs.clamp(0, 1))
        
        adv_inputs.grad.zero_()
    flipped_bits = ((outputs > 0.5).float() - original_hash).abs()

    clean_PIL_images = [T.ToPILImage()(img) for img in inputs]
    adv_PIL_images = [T.ToPILImage()(img) for img in adv_inputs]

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
parser.add_argument('--alpha', dest='alpha', type=float, default=1/255,
                    help='step size for each iteration')
parser.add_argument('--num_iter', dest='num_iter', type=int, default=50,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')

args = parser.parse_args()

image_files = [join(args.image_dir, f) for f in os.listdir(args.image_dir) if isfile(join(args.image_dir, f))]

batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]

for batch in batches:
    pgd_attack(batch, alpha=args.alpha, num_iter=args.num_iter, epsilon=args.epsilon)
    exit()