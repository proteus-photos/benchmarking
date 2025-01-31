import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
import argparse
from scipy.stats import binom
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from transformer import Transformer
from database import Database
import torch

from hashes.dinohash import dinohash, preprocess, dinov2
from hashes.neuralhash import neuralhash, preprocess

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:10_000]
tensors = torch.stack([preprocess(Image.open(os.path.join(dataset_folder, f)).convert("RGB")) for f in image_files]).numpy()

BATCH_SIZE = 64
@torch.no_grad()
def f():
    for i in tqdm(range(0, len(tensors), BATCH_SIZE)):
        batch = tensors[i:i+BATCH_SIZE]
        neuralhash(batch)
        del batch