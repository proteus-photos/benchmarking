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

load_and_preprocess_img = lambda img: preprocess(Image.open(img).convert('RGB')).cuda().unsqueeze(0)
import torch
import torch.nn as nn
import torch.optim as optim

def cw_attack_batch(model, images, confidence=0, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    """
    Batched implementation of the C&W L2 attack.
    
    Args:
        model: The target model to attack
        images: Batch of input images tensor (N, C, H, W)
        confidence: Confidence parameter (higher -> stronger attack)
        c: Initial c value for the attack
        kappa: Kappa parameter from the paper
        max_iter: Maximum number of optimization iterations
        learning_rate: Learning rate for optimization
    
    Returns:
        perturbed_images: Batch of adversarial examples
        success_mask: Boolean mask indicating which attacks succeeded
    """
    model.eval()
    batch_size = images.shape[0]
    
    w = torch.zeros_like(images, requires_grad=True)
    optimizer = optim.Adam([w], lr=learning_rate)
    
    for step in range(max_iter):
        if not active_mask.any():
            break
            
        perturbed = torch.tanh(w) * 0.5 + 0.5
        
        l2_dists = torch.norm((perturbed - images).view(batch_size, -1), p=2, dim=1)
        
        outputs = model(normalize(perturbed))
        
        # Calculate CW loss only for active images
        losses = torch.clamp(max_other_scores - target_scores + confidence, min=0)
        total_loss = (l2_dists + c * losses)[active_mask].mean()
        
        # Update weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Check which attacks succeeded
        _, predicted = torch.max(outputs.data, 1)
        current_success = (predicted == targets)
        
        # Update best adversarial examples
        improved = (current_success & (l2_dists < best_l2s))
        best_advs[improved] = perturbed[improved].clone()
        best_l2s[improved] = l2_dists[improved].clone()
        
        # Update success mask
        success_mask = success_mask | current_success
        
        # Update active mask - only keep trying for unsuccessful attacks
        active_mask = ~success_mask
        
        # Optional: Print progress
        if step % 100 == 0:
            print(f"Step {step}: {success_mask.sum().item()}/{batch_size} images successfully attacked")
    
    # Return best adversarial examples found for each image
    final_advs = torch.where(success_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                            best_advs,
                            perturbed)
    
    return final_advs, success_mask

def test_attack_batch(model, images, target_classes):
    """
    Test the batched C&W attack
    """
    # Convert inputs to tensors if needed
    if not isinstance(images, torch.Tensor):
        images = torch.FloatTensor(images)
    if not isinstance(target_classes, torch.Tensor):
        target_classes = torch.LongTensor(target_classes)
        
    # Run attack
    perturbed_images, success_mask = cw_attack_batch(model, images, target_classes)
    
    # Print results
    success_rate = success_mask.float().mean().item()
    print(f"Attack success rate: {success_rate:.2%}")
    
    # Calculate L2 distances for successful attacks
    l2_dists = torch.norm((perturbed_images - images).view(len(images), -1), p=2, dim=1)
    successful_l2 = l2_dists[success_mask]
    
    if len(successful_l2) > 0:
        print(f"Mean L2 distance for successful attacks: {successful_l2.mean():.4f}")
        
    return perturbed_images, success_mask

# Example usage:
"""
model = YourModel()
images = your_batch_of_images  # Shape: (N, C, H, W)
target_classes = desired_targets  # Shape: (N,)

adversarial_examples, success_mask = test_attack_batch(model, images, target_classes)
"""

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Perform neural collision attack.')
    parser.add_argument('--source', dest='source', type=str,
                        default='inputs/source.png', help='image to manipulate')
    parser.add_argument('--learning_rate', dest='learning_rate', default=1e-3,
                        type=float, help='step size of PGD optimization step')
    parser.add_argument('--optimizer', dest='optimizer', default='Adam',
                        type=str, help='kind of optimizer')
    parser.add_argument('--ssim_weight', dest='ssim_weight', default=5,
                        type=float, help='weight of ssim loss')
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
    parser.add_argument('--threads', dest='num_threads',
                        default=4, type=int, help='Number of parallel threads')
    parser.add_argument('--check_interval', dest='check_interval',
                        default=1, type=int, help='Hash change interval checking')
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

    threads_args = [(images, device, seed, loss_function,
                     logger, args, pbar) for i in range(args.num_threads)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        executor.map(thread_function, threads_args)

    logger.finish_logging()
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()
