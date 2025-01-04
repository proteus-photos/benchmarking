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

def project(x_adv, x0, epsilon):
    return x_adv.clamp(x0-epsilon, x0+epsilon).clamp(0, 1)

def pgd_attack(images, alpha=2/255, num_iter=50, epsilon=8/255, w=[0, 0.22, 0.41, 0.57, 0.7, 0.8, 0.87, 0.93, 0.99]):
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
    w = np.array(w)
    w_int = np.ceil(w * num_iter).astype(int)
    halved = np.zeros_like(w_int)
    
    inputs = torch.stack([preprocess(Image.open(img).convert('RGB')).cuda() for img in images])
    original_hash = (dinohash(inputs, differentiable=True) > 0.5).float()
    
    x_best = inputs.clone()

    outputs = dinohash(x, differentiable=True, c=5)
    loss_best = torch.nn.functional.mse_loss(outputs, original_hash, reduction="mean").mean(1)

    for k in range(num_iter):
        x.requires_grad_()
        # Forward pass
        outputs = dinohash(x, differentiable=True, c=5)
        loss = torch.nn.functional.mse_loss(outputs, original_hash).mean(1)
        loss.backward()
        grad = x.grad.detach()

        with torch.no_grad():
            z = project(x + alpha * grad, inputs, epsilon)
            x = x + alpha * (z - x) + (1-alpha) * (x - x_prev)
            x = project(x, inputs, epsilon)

        loss = torch.nn.functional.mse_loss(dinohash(x, differentiable=True, c=5), original_hash).mean(1)
        
        improved = loss > loss_best
        x_best[improved] = x[improved]
        loss_best[improved] = loss[improved]

        if k in w:
            pass
            
    
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