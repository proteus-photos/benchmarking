import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import gc
import argparse
from scipy.stats import binom
import pandas as pd

from transformer import Transformer
from database import Database

from hashes.dinohash import dinohash


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

hasher = dinohash # dhash, phash, blockhash, whash

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:1_000_000]

BATCH_SIZE = 512
N_IMAGE_RETRIEVAL = 1

parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking.')
parser.add_argument('-r', '--refresh', action='store_true')
parser.add_argument('--transform')
args = parser.parse_args()

transformation = args.transform
t = Transformer()

os.makedirs("databases", exist_ok=True)
if hasher.__name__ + ".npy" not in os.listdir("databases") or args.refresh:
    print("Creating database for", hasher.__name__)
    original_hashes = []
    image_file_batches = (image_files[i:i+BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE))

    for image_file_batch in tqdm(image_file_batches, total=len(image_files)//BATCH_SIZE):
        images = [Image.open(os.path.join(dataset_folder, image_file)).convert("RGB") for image_file in image_file_batch]
        original_hashes.extend(hasher(images, defense=False))
        gc.collect()
        
    db = Database(original_hashes, storedir=f"databases/{hasher.__name__}")
else:
    db = Database(None, storedir=f"databases/{hasher.__name__}")

### Evaluate Hamming distance

# transformed_images = [t.transform(Image.open(os.path.join(dataset_folder, image_file)).convert("RGB"), transformation) for image_file in tqdm(image_files)]
# hashes = hasher(transformed_images)
# not_hashes = hashes[::-1]

# matches = db.similarity_score(hashes)
# print("True:")
# print(matches.mean(), matches.std())

# not_matches = db.similarity_score(not_hashes)
# print("False:")
# print(not_matches.mean(), not_matches.std())

# n_matches = 0
# print(f"Computing top {N_IMAGE_RETRIEVAL} accuracy...")
# for index, image_file in enumerate(tqdm(image_files)):
#     image = Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")
#     transformed_image = t.transform(image, transformation)
#     modified_hash = hasher([transformed_image])[0]
#     result = db.query(modified_hash, k=N_IMAGE_RETRIEVAL)
#     if index in [point["index"] for point in result]:
#         n_matches += 1

#     gc.collect()

print(f"Computing bit accuracy for {transformation} + {hasher.__name__}...")
modified_hashes = []

image_file_batches = (image_files[i:i+BATCH_SIZE] for i in range(0, len(image_files), BATCH_SIZE))

for image_file_batch in tqdm(image_file_batches, total=len(image_files)//BATCH_SIZE):
    images = [Image.open(os.path.join(dataset_folder, image_file)).convert("RGB") for image_file in image_file_batch]
    transformed_images = [t.transform(t.transform(image, "screenshot"), transformation) for image in images]
    modified_hashes_batch = hasher(transformed_images, defense=False).tolist()
    modified_hashes.extend(modified_hashes_batch)

modified_hashes = np.array(modified_hashes)
bits = modified_hashes.shape[-1]

matches = db.similarity_score(modified_hashes)
inv_matches = db.similarity_score(modified_hashes[::-1])

print(matches.mean(), matches.std())
print(inv_matches.mean(), inv_matches.std())

generate_roc(matches, bits=bits)

