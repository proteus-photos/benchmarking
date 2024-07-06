import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from transformer import Transformer
from blockhash.blockhash import blockhash
from neuralhash.neuralhash import neuralhash
from utils import match
from database import Database

from colourhash.dhash import dhash
from colourhash.ahash import ahash
from colourhash.phash import phash
from colourhash.whash import whash

transformations = ['jpeg', 'crop'] #, 'screenshot', 'double screenshot']
hash_methods = [neuralhash] # dhash, blockhash, ahash, phash, whash]

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)]

t = Transformer()

transformed_images_list = [[] for transformation in transformations]
for image_file in image_files:
    image = Image.open(os.path.join(dataset_folder, image_file))
    if image.mode == "L":
        continue
    for i, transformation in enumerate(transformations):            
        transformed_image = t.transform(image, transformation)
        transformed_images_list[i].append(transformed_image)

os.makedirs("databases", exist_ok=True)
databases = []

for hash_method in hash_methods:
    if hash_method.__name__ + ".npy" not in os.listdir("databases"):
        original_hashes = hash_method(images)
        db = Database(original_hashes, storedir=f"databases/{hash_method.__name__}")
    else:
        db = Database(None, storedir=f"databases/{hash_method.__name__}")

    databases.append(db)

n_matches = np.zeros((len(hash_methods), len(transformations)))


for i, (hash_method, database) in enumerate(zip(hash_methods, databases)):
    for j, transformed_images in tqdm(enumerate(transformed_images_list)):
        modified_hashes = hash_method(transformed_images)
        for index, modified_hash in enumerate(modified_hashes):
            result = database.query(modified_hash, k=5)
            if index in [point["index"] for point in result]:
                n_matches[i, j] += 1

for i, hash_method in enumerate(hash_methods):
    for j, transformation in enumerate(transformations):
        print(f'{hash_method.__name__} with {transformation} transformation:', n_matches[i, j] / len(images))

    print("#############################################")