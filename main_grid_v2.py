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
from utils import match, create_model, transform, reparametricize, chunk_call, tilize_by_anchors
from database import TileDatabaseV2

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash
from hashes.neuralhash import neuralhash #, initialize_session

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

transformation = 'screenshot' #, 'double screenshot', 'jpeg', 'crop']
hasher = neuralhash # dhash, phash, blockhash, whash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)][:1_000]

N_IMAGE_RETRIEVAL = 5
N_BREAKS = 2
MIN_MARGIN = 0.4
parser = argparse.ArgumentParser(description ='Perform retrieval benchmarking based on grids.')
parser.add_argument('-r', '--refresh', action='store_true')

args = parser.parse_args()

t = Transformer()
model = create_model("finetuned_mobilenetv3.pth", backbone="mobilenet_old")
model.eval()
images = [copy.deepcopy(Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")) for image_file in image_files]

os.makedirs("tile_databases_v2", exist_ok=True)
if hasher.__name__ + ".npy" not in os.listdir("tile_databases") or args.refresh:
    # initialize_session()
    print("Creating database for", hasher.__name__)
    original_hashes = []
    print("meow")
    image_arrays = [transform(image) for image in tqdm(images)]
    print("meow2")
    anchor_points_list = chunk_call(model, torch.stack(image_arrays), 256)
    print("meow3")
    anchor_points_list = reparametricize(anchor_points_list, MIN_MARGIN).numpy()

    # anchor_points_list[:, 0] = 0.1
    # anchor_points_list[:, 1] = 0.1
    # anchor_points_list[:, 2] = 0.9
    # anchor_points_list[:, 3] = 0.9

    # each image has a different grid size of hashes
    original_hashes = []
    n_ranges = []
    for image, anchor_points in zip(tqdm(images), anchor_points_list):
        tiles, n_range = tilize_by_anchors(image, N_BREAKS, anchor_points)
        grid_shape = (n_range[Y2] - n_range[Y1], n_range[X2] - n_range[X1])
        tile_hashes = hasher(tiles).reshape(*grid_shape, -1)
        
        original_hashes.append(tile_hashes)
        n_ranges.append(n_range)
    metadata = {"anchors": anchor_points_list.tolist(), "n_ranges": n_ranges}
    database = TileDatabaseV2(original_hashes, storedir=f"tile_databases_v2/{hasher.__name__}", metadata=metadata, n_breaks=N_BREAKS)
else:
    database = TileDatabaseV2(None, storedir=f"tile_databases_v2/{hasher.__name__}", n_breaks=N_BREAKS)


print("Computing top 5 accuracy...")
# crop_sizes = np.random.uniform(0, 0.1, (len(images, 4)))
# crop_sizes[:, 2] = 1 - crop_sizes[:, 2]
# crop_sizes[:, 3] = 1 - crop_sizes[:, 3]

transformed_images = [t.transform(image, transformation) for image in images]
anchor_points_list = chunk_call(model, torch.stack([transform(image) for image in transformed_images]), 256)
anchor_points_list = reparametricize(anchor_points_list, MIN_MARGIN).numpy()

# anchor_points_list[:, 0] = 0
# anchor_points_list[:, 1] = 0
# anchor_points_list[:, 2] = 1
# anchor_points_list[:, 3] = 1

chunks_of = 100
transformed_images_chunks = [transformed_images[i:i+chunks_of] for i in range(0, len(images), chunks_of)]
anchor_points_chunks = [anchor_points_list[i:i+chunks_of] for i in range(0, len(anchor_points_list), chunks_of)]
print("starting....")
with Pool(processes=5,) as p: #  initializer=initialize_session
    queries = list(tqdm(p.imap(my_starmap,
                        [(transformed_images_chunk, hasher, anchor_points_chunk, N_IMAGE_RETRIEVAL)
                        for transformed_images_chunk, anchor_points_chunk in
                        zip(transformed_images_chunks, anchor_points_chunks)]), total=len(transformed_images_chunks)))

results = [point for points in queries for point in points]

n_matches = 0
for index, result in enumerate(results):
    match = index in [point["index"] for point in result]
    if match:
        n_matches += 1
    else:
        print("failed at", index, "with score", result)

n_matches = sum([int(index in [point["index"] for point in result]) for index, result in enumerate(results)])
# for index, result in enumerate(results):
#     n_matches += int(index in [point["index"] for point in result])

print(f'{hasher.__name__} with {transformation} transformation:', n_matches / len(image_files))
print("#############################################")