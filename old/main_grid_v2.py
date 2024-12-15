import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import argparse
import copy
from multiprocessing import Pool
from random import random

import torch

from transformer import Transformer
from utils import match, create_model, transform, reparametricize, chunk_call, tilize_by_anchors
from database import TileDatabaseV2

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash
# from hashes.neuralhash import neuralhash #, initialize_session
from hashes.dinohash import dinohash

"""
Gridded version but with grid aligned to anchor points
"""

X = 0
Y = 1
W = 2
H = 3

X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

def my_starmap(x):
    return database.multi_query(*x)

def evaluate_model(image):
    image_array = transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        anchor_points = model(image_array).cpu()
        anchor_points = reparametricize(anchor_points, MIN_MARGIN).numpy()[0]
        
    del image_array
    return anchor_points

transformation = 'screenshot'
hasher = dinohash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)][:50_000]

N_IMAGE_RETRIEVAL = 5
N_BREAKSS = [1]
print(N_BREAKSS)
MIN_MARGIN = 0.4
N_TRANSFORMS = 1

parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking based on grids.')
parser.add_argument('-r', '--refresh', action='store_true')

args = parser.parse_args()

t = Transformer()
# model = create_model("finetuned_mobilenetv3.pth", backbone="mobilenet_old")
# model.eval()

model = None

original_images = [copy.deepcopy(Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")) for image_file in image_files]

images = []
for original_image in tqdm(original_images):
    images.append(original_image)
    for i in range(N_TRANSFORMS - 1):
        horizontal = random() * 0.2
        vertical = random() * 0.2

        left = horizontal * random()
        top = vertical * random()
        right = 1 - (horizontal - left)
        bottom = 1 - (vertical - top)

        transformed_image = t.transform(original_image, "crop", left=left, top=top, right=right, bottom=bottom)
        images.append(transformed_image)

os.makedirs("tile_databases_v2", exist_ok=True)
print(os.listdir("tile_databases_v2"))
if hasher.__name__ + "_hashes.pkl" not in os.listdir("tile_databases_v2") or args.refresh:
    # initialize_session()
    print("Creating database for", hasher.__name__)
    original_hashes = []
    image_arrays = [transform(image) for image in tqdm(images)]
    anchor_points_list = chunk_call(model, torch.stack(image_arrays), 256)
    anchor_points_list = reparametricize(anchor_points_list, MIN_MARGIN).numpy()
    anchor_points_list = np.repeat(anchor_points_list, len(N_BREAKSS), axis=0)

    del image_arrays

    # each image has a different grid size of hashes
    original_hashes = []
    n_ranges = []
    
    for image, anchor_points in zip(tqdm(images), anchor_points_list):
        for N_BREAKS in N_BREAKSS:
            tiles, n_range = tilize_by_anchors(image, N_BREAKS, anchor_points)
            grid_shape = (n_range[Y2] - n_range[Y1], n_range[X2] - n_range[X1])
            tile_hashes = hasher(tiles).reshape(*grid_shape, -1)
            
            original_hashes.append(tile_hashes)
            n_ranges.append(n_range)
    
    indexes = np.arange(len(original_images))
    indexes = np.repeat(indexes, N_TRANSFORMS * len(N_BREAKSS))

    metadata = {"anchors": anchor_points_list.tolist(), "n_ranges": n_ranges, "indices": indexes.tolist()}
    database = TileDatabaseV2(original_hashes, storedir=f"tile_databases_v2/{hasher.__name__}", metadata=metadata, n_breaks=N_BREAKSS)
else:
    database = TileDatabaseV2(None, storedir=f"tile_databases_v2/{hasher.__name__}", n_breaks=N_BREAKSS)

print(f"Computing top {N_IMAGE_RETRIEVAL} accuracy...")

transformed_images = [t.transform(image, transformation) for image in tqdm(original_images)]
anchor_points_list = chunk_call(model, torch.stack([transform(image) for image in transformed_images]), 256)
anchor_points_list = reparametricize(anchor_points_list, MIN_MARGIN).numpy()

### Evaluation for true hamming distance
matches = []
inv_matches = []
for index, (transformed_image, anchor_points) in enumerate(zip(tqdm(transformed_images), anchor_points_list)):
    match = database.similarity_score(transformed_image, hasher, anchor_points, index, flexible=True)["matches"]
    inv_match = database.similarity_score(transformed_image, hasher, anchor_points, len(transformed_images)-1-index, flexible=True)["matches"]

    matches.append(max(point["score"] for point in match))
    inv_matches.append(max(point["score"] for point in inv_match))

matches = np.array(matches)
inv_matches = np.array(inv_matches)
while True:
    tau = input("Enter tau: ")
    print("fnr:", np.mean(matches < float(tau)))
    print("fpr:", np.mean(inv_matches > float(tau)))

print(N_BREAKSS)
print(matches.mean(), matches.std())

# results = []
# for transformed_image, anchor_points in zip(tqdm(transformed_images), anchor_points_list):
#     result = database.query(transformed_image, hasher, anchor_points, K_RETRIEVAL=N_IMAGE_RETRIEVAL)
#     results.append(result)

## Multiprocessing

# chunks_of = 20
# transformed_images_chunks = [transformed_images[i:i+chunks_of] for i in range(0, len(images), chunks_of)]
# anchor_points_chunks = [anchor_points_list[i:i+chunks_of] for i in range(0, len(anchor_points_list), chunks_of)]

# print("starting....")
# with Pool(processes=5,) as p:
#     queries = list(tqdm(p.imap(my_starmap,
#                         [(transformed_images_chunk, hasher, anchor_points_chunk, N_IMAGE_RETRIEVAL)
#                         for transformed_images_chunk, anchor_points_chunk in
#                         zip(transformed_images_chunks, anchor_points_chunks)]), total=len(transformed_images_chunks)))

# results = [point for points in queries for point in points]


## Evaluation with failures
# n_matches = 0
# for index, result in enumerate(results):
#     match = index in [point["index"] for point in result["matches"]]
#     if match:
#         n_matches += 1
#     else:
#         print("failed at", index, "with score", result)

### Evaluation without failures
# n_matches = sum([int(index in [point["index"] for point in result["matches"]]) for index, result in enumerate(results)])

print(f'{hasher.__name__} with {transformation} transformation:', n_matches / len(image_files))
print("#############################################")