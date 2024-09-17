import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import argparse
import copy
from multiprocessing import Pool

import torch

from transformer import Transformer
from utils import match, tilize, create_model, transform, reparametricize, chunk_call
from database import TileDatabase

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash
from hashes.neuralhash import neuralhash#, initialize_session

def my_starmap(x):
    return database.multi_query(*x)
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
    # initialize_session()
    print("Creating database for", hash_method.__name__)
    original_hashes = []
    image_arrays = [transform(image) for image in images]
    anchor_points_list = chunk_call(model, torch.stack(image_arrays), 256)
    anchor_points_list = reparametricize(anchor_points_list, 0.5).numpy()

    # anchor_points_list[:, 0] = 0.1
    # anchor_points_list[:, 1] = 0.1
    # anchor_points_list[:, 2] = 0.9
    # anchor_points_list[:, 3] = 0.9

    original_hashes = [tile_hash for image in tqdm(images) for tile_hash in hash_method(tilize(image, N_TILES))]

    anchor_points_list = np.array(anchor_points_list)
    original_hashes = np.array(original_hashes).reshape(len(image_files), N_TILES, N_TILES, -1)
    database = TileDatabase(N_TILES, original_hashes, storedir=f"tile_databases/{hash_method.__name__}", anchors=anchor_points_list)
else:
    database = TileDatabase(N_TILES, None, storedir=f"tile_databases/{hash_method.__name__}")


print("Computing top 5 accuracy...")
# crop_sizes = np.random.uniform(0, 0.1, (len(images, 4)))
# crop_sizes[:, 2] = 1 - crop_sizes[:, 2]
# crop_sizes[:, 3] = 1 - crop_sizes[:, 3]

transformed_images = [t.transform(image, transformation) for image in images]
anchor_points_list = chunk_call(model, torch.stack([transform(image) for image in transformed_images]), 256)
anchor_points_list = reparametricize(anchor_points_list, 0.5).numpy()

# anchor_points_list[:, 0] = 0
# anchor_points_list[:, 1] = 0
# anchor_points_list[:, 2] = 1
# anchor_points_list[:, 3] = 1

chunks_of = 20
transformed_images_chunks = [transformed_images[i:i+chunks_of] for i in range(0, len(images), chunks_of)]
anchor_points_chunks = [anchor_points_list[i:i+chunks_of] for i in range(0, len(anchor_points_list), chunks_of)]
print("starting....")
with Pool(processes=5,) as p: #  initializer=initialize_session
    queries = list(tqdm(p.imap(my_starmap,
                        [(transformed_images_chunk, hash_method, anchor_points_chunk, N_IMAGE_RETRIEVAL)
                        for transformed_images_chunk, anchor_points_chunk in
                        zip(transformed_images_chunks, anchor_points_chunks)]), total=len(transformed_images_chunks)))

results = [point for points in queries for point in points]

n_matches = 0
for index, result in enumerate(results):
    if index in [point["index"] for point in result]:
        n_matches += 1

print(f'{hash_method.__name__} with {transformation} transformation:', n_matches / len(image_files))
print("#############################################")