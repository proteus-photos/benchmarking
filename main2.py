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

from hashes.neuralhash import neuralhash, preprocess

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(dataset_folder, self.image_files[idx])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return preprocess(image)

def combined_transform(image):
    transformations = ["screenshot", transformation, "erase", "text"]
    for transform in transformations:
        image = t.transform(image, method=transform)
    return image

def generate_roc(matches, bits):
    matches = matches * bits
    taus = np.arange(bits+1)
    tpr = [(matches>=tau).mean() for tau in taus]

    fpr = 1 - binom.cdf(taus-1, bits, 0.5)
    
    df = pd.DataFrame({
        "tpr": tpr,
        "fpr": fpr,
        "tau": taus
    })
    
    df.to_csv(f"./results/{hasher.__name__}_{transformation}.csv")

hasher = neuralhash

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:1_000_000]

BATCH_SIZE = 64
N_IMAGE_RETRIEVAL = 1

parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking.')
parser.add_argument('-r', '--refresh', action='store_true')
parser.add_argument('--transform')
args = parser.parse_args()

transformation = args.transform
t = Transformer()

dataset = ImageDataset(image_files, transform=combined_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

os.makedirs("databases", exist_ok=True)
if hasher.__name__ + ".npy" not in os.listdir("databases") or args.refresh:
    print("Creating database for", hasher.__name__)
    original_hashes = []
    image_file_batches = (image_files[i:i+BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE))

    for image_batch in tqdm(dataloader):
        original_hashes.extend(hasher(image_batch).cpu())
        gc.collect()
        
    db = Database(original_hashes, storedir=f"databases/{hasher.__name__}")
else:
    db = Database(None, storedir=f"databases/{hasher.__name__}")

print(f"Computing bit accuracy for {transformation} + {hasher.__name__}...")
modified_hashes = []

for transformed_images in tqdm(dataloader):
    modified_hashes_batch = hasher(transformed_images).tolist()
    modified_hashes.extend(modified_hashes_batch)

modified_hashes = np.array(modified_hashes)
bits = modified_hashes.shape[-1]

matches = db.similarity_score(modified_hashes)
inv_matches = db.similarity_score(modified_hashes[::-1])

print(matches.mean(), matches.std())
print(inv_matches.mean(), inv_matches.std())

with open(f"./results/{hasher.__name__}_{transformation}.txt", "w") as f:
    f.write(f"Bit accuracy: {matches.mean()} / {matches.std()}\n")
    f.write(f"Random accuracy: {inv_matches.mean()} / {inv_matches.std()}\n")

generate_roc(matches, bits=bits)

