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


def pgd_attack(model, images, alpha, num_iter=50, epsilon=1/8):
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
    original_hash = dinohash(inputs, differentiable=True) > 0.5
    adv_inputs = inputs.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        # Forward pass
        outputs = dinohash(adv_inputs, differentiable=True, c=5)

        loss = -torch.nn.functional.mse_loss(outputs, original_hash) / original_hash.shape[1]
        loss.backward()
        
        with torch.no_grad():
            adv_inputs = adv_inputs + alpha * adv_inputs.grad.sign()
            
            # project perturbations to L∞ norm ball
            adv_inputs = torch.min(torch.max(adv_inputs, inputs - epsilon), inputs + epsilon)
            adv_inputs = torch.clamp(adv_inputs, 0, 1)
        
        adv_inputs.grad.zero_()

    adv_PIL_images = [T.ToPILImage()(img) for img in adv_inputs]
    clean_PIL_images = [T.ToPILImage()(img) for img in inputs]
    
    for img, adv_img in zip(PIL_images):
        img.save(f'./temp/{i}.png')

    return adv_inputs.detach()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='change_hash_attack', type=str, help='name of the experiment and logging file')
    parser.add_argument('--output_folder', dest='output_folder',
                        default='evasion_attack_outputs', type=str, help='folder to save optimized images in')
    parser.add_argument('--edges_only', dest='edges_only',
                        action='store_true', help='Change only pixels of edges')
    parser.add_argument('--optimize_original', dest='optimize_original',
                        action='store_true', help='Optimize resized image')
    parser.add_argument('--hamming', dest='hamming',
                        default=0.4, type=float, help='Minimum Hamming distance to stop')
    args = parser.parse_args()

    # Create temp folder
    os.makedirs('./temp', exist_ok=True)

    # Load and prepare components
    start = time.time()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = load_hash_matrix()
    seed = torch.tensor(seed).to(device)

    # Prepare output folder
    if args.output_folder != '':
        try:
            os.mkdir(args.output_folder)
        except:
            if not os.listdir(args.output_folder):
                print(
                    f'Folder {args.output_folder} already exists and is empty.')
            else:
                print(
                    f'Folder {args.output_folder} already exists and is not empty.')

    # Prepare logging
    logging_header = ['file', 'optimized_file', 'l2',
                      'l_inf', 'ssim', 'steps', 'target_loss', 'Hamming']
    logger = Logger(args.experiment_name, logging_header, output_dir='./logs')
    logger.add_line(['Hyperparameter', args.source, args.learning_rate,
                     args.optimizer, args.ssim_weight, args.edges_only, args.hamming])

    # define loss function
    loss_function = mse_loss

    # Load images
    if os.path.isfile(args.source):
        images = [args.source]
    elif os.path.isdir(args.source):
        images = [join(args.source, f) for f in os.listdir(
            args.source) if isfile(join(args.source, f))]
        images = sorted(images)
    else:
        raise RuntimeError(f'{args.source} is neither a file nor a directory.')

    # Start threads
    def thread_function(x): return optimization_thread( images, device, seed, loss_function, logger, args, pbar)
    
    pbar = tqdm(total=len(images), desc="Processing")

    logger.finish_logging()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
