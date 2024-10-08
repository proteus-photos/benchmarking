import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import argparse

from transformer import Transformer
from hashes.blockhash import blockhash
# from hashes.neuralhash import neuralhash
from utils import match
from database import Database

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash
from hashes.neuralhash import neuralhash

transformation = 'screenshot' #, 'double screenshot', 'jpeg', 'crop']
hasher = neuralhash # dhash, phash, blockhash, whash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)][:1000]

N_IMAGE_RETRIEVAL = 5

parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking.')
parser.add_argument('-r', '--refresh', action='store_true')

args = parser.parse_args()

t = Transformer()

os.makedirs("databases", exist_ok=True)
if hasher.__name__ + ".npy" not in os.listdir("databases") or args.refresh:
    print("Creating database for", hasher.__name__)
    original_hashes = []
    for image_file in tqdm(image_files):
        image = Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")
        original_hashes.append(hasher([image])[0])
        gc.collect()
    db = Database(original_hashes, storedir=f"databases/{hasher.__name__}")
else:
    db = Database(None, storedir=f"databases/{hasher.__name__}")

### Evaluate Hamming distance

transformed_images = [t.transform(Image.open(os.path.join(dataset_folder, image_file)).convert("RGB"), transformation) for image_file in image_files]
hashes = hasher(transformed_images)
not_hashes = hashes[::-1]

matches = db.similarity_score(hashes)
print("True:")
print(matches.mean(), matches.std())

not_matches = db.similarity_score(not_hashes)
print("False:")
print(not_matches.mean(), not_matches.std())

# n_matches = 0
# print("Computing top 5 accuracy...")
# for index, image_file in enumerate(tqdm(image_files)):
#     image = Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")
#     transformed_image = t.transform(image, transformation)
#     modified_hash = hasher([transformed_image])[0]
#     result = db.query(modified_hash, k=N_IMAGE_RETRIEVAL)
#     if index in [point["index"] for point in result]:
#         n_matches += 1

#     gc.collect()
# print(f'{hasher.__name__} with {transformation} transformation:', n_matches / len(image_files))
