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

def hash_loss_grad(x, original_hash):
    x.requires_grad = True
    hash = dinohash(x, differentiable=True, c=5)

    # contains the loss for each image in the batch
    loss = torch.nn.functional.mse_loss(hash, original_hash).mean(1)

    # contains overall sum of loss for batch, we dont use mean
    loss_sum = loss.sum()

    loss_sum.backward()
    grad = x.grad.detach()

    x.requires_grad = False

    return hash, loss, grad

def project(x_adv, x0, epsilon):
    return x_adv.clamp(x0-epsilon, x0+epsilon).clamp(0, 1)

def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).cuda()
    for counter5 in range(k):
        t += (x[j - counter5] > x[j - counter5 - 1]).float()
    return (t <= k * k3 * torch.ones_like(t)).float()

def pgd_attack(images, num_iter=50, epsilon=8/255, w=[0, 0.22, 0.41, 0.57, 0.7, 0.8, 0.87, 0.93, 0.99]):
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
    rho = 0.75
    num_iter_2 = max(int(0.22 * num_iter), 1)
    num_iter_min = max(int(0.06 * num_iter), 1)
    size_decr = max(int(0.03 * num_iter), 1)

    w = np.array(w)
    w_int = np.ceil(w * num_iter).astype(int)
    halved = np.zeros_like(w_int)
    
    inputs = torch.stack([preprocess(Image.open(img).convert('RGB')).cuda() for img in images])
    original_hash = (dinohash(inputs, differentiable=True) > 0.5).float()
    
    x_adv = inputs.clone()
    x_best = inputs.clone()
    loss_steps = torch.zeros(num_iter, inputs.size(0)).to(inputs.device)
    loss_best_steps = torch.zeros(num_iter+1, inputs.size(0)).to(inputs.device)

    x_adv.requires_grad_()
    hash, loss, grad = hash_loss_grad(x_adv, original_hash)
    grad_best = grad.clone()
    loss_best = loss.detach().clone()

    alpha = 2.

    step_size = alpha * epsilon * torch.ones(inputs.size(0), 3).to(inputs.device).detach()
    x_adv_old = x_adv.clone()

    counter3 = 0
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)

    for k in range(num_iter):
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if k > 0 else 1.0

            x_adv_1 = project(x_adv + step_size * grad.sign(), inputs, epsilon)
            x_adv_1 = project(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a))
            
            x_adv = x_adv_1 + 0.

        x_adv.requires_grad_()
        hash, loss, grad = hash_loss_grad(x_adv, original_hash)
        
        with torch.no_grad():
            y1 = loss.detach().clone()
            loss_steps[k] = y1.clone()
            ind_improved = (y1 > loss_best).nonzero().squeeze()
            x_best[ind_improved] = x_adv[ind_improved].clone()
            grad_best[ind_improved] = grad[ind_improved].clone()
            loss_best[ind_improved] = y1[ind_improved].clone()
            loss_best_steps[k+1] = loss_best.clone()

            counter3 += 1
            if counter3==num_iter_2:
                fl_oscillation = check_oscillation(loss_steps, i, k,
                    loss_best, k3=rho)
                fl_reduce_no_impr = (1. - reduced_last_check) * (
                    loss_best_last_check >= loss_best).float()
                fl_oscillation = torch.max(fl_oscillation, fl_reduce_no_impr)
                reduced_last_check = fl_oscillation.clone()
                loss_best_last_check = loss_best.clone()

                if fl_oscillation.sum() > 0:
                    ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                    step_size[ind_fl_osc] /= 2.0

                    x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                    grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                k = max(k - size_decr, num_iter_min) ##FIXME: check this

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