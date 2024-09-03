import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import argparse
import copy

import torch

from transformer import Transformer
from utils import match, tilize, create_model, transform, reparametricize, chunk_call
from database import TileDatabase

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash
from hashes.neuralhash import neuralhash

def evaluate_model(image):
    image_array = transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        anchor_points = model(image_array).cpu()
        anchor_points = reparametricize(anchor_points, 0.5).numpy()[0]
        
    del image_array
    return anchor_points

transformation = 'screenshot' #, 'double screenshot', 'jpeg', 'crop']
hash_method = neuralhash # dhash, phash, blockhash, whash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)][:1_000]

N_IMAGE_RETRIEVAL = 5
N_TILES = 7
parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking based on grids.')
parser.add_argument('-r', '--refresh', action='store_true')

args = parser.parse_args()

t = Transformer()
model = create_model("finetuned_mobilenetv3.pth")
model.eval()
images = [copy.deepcopy(Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")) for image_file in image_files]

os.makedirs("tile_databases", exist_ok=True)
if hash_method.__name__ + ".npy" not in os.listdir("tile_databases") or args.refresh:
    print("Creating database for", hash_method.__name__)
    original_hashes = []
    image_arrays = [transform(image) for image in tqdm(images)]
    anchor_points_list = chunk_call(model, torch.stack(image_arrays), 256)
    original_hashes = [tile_hash for image in tqdm(images) for tile_hash in hash_method(tilize(image, N_TILES))]

    anchor_points_list = np.array(anchor_points_list)
    original_hashes = np.array(original_hashes).reshape(len(image_files), N_TILES, N_TILES, -1)
    database = TileDatabase(N_TILES, original_hashes, storedir=f"tile_databases/{hash_method.__name__}", anchors=anchor_points_list)
else:
    database = TileDatabase(N_TILES, None, storedir=f"tile_databases/{hash_method.__name__}")

n_matches = 0

print("Computing top 5 accuracy...")
for index, image in enumerate(tqdm(images)):
    transformed_image = t.transform(image, transformation)
    anchor_points = evaluate_model(transformed_image)
    result = database.query(transformed_image, hash_method, anchor_points, k=N_IMAGE_RETRIEVAL)
    if index in [point["index"] for point in result]:
        n_matches += 1

    gc.collect()

print(f'{hash_method.__name__} with {transformation} transformation:', n_matches / len(image_files))
print("#############################################")