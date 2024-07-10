import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc

from transformer import Transformer
from hashes.blockhash import blockhash
from hashes.neuralhash import neuralhash
from utils import match
from database import Database

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash

transformations = ['screenshot'] #, 'double screenshot', 'jpeg', 'crop']
hash_methods = [phash, neuralhash] #, blockhash, whash, dhash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)]

t = Transformer()

os.makedirs("databases", exist_ok=True)
databases = []

for hash_method in hash_methods:
    if hash_method.__name__ + ".npy" not in os.listdir("databases"):
        print("Creating database for", hash_method.__name__)
        original_hashes = []
        for image_file in tqdm(image_files):
            image = Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")
            original_hashes.append(hash_method([image])[0])
            gc.collect()
        db = Database(original_hashes, storedir=f"databases/{hash_method.__name__}")
    else:
        db = Database(None, storedir=f"databases/{hash_method.__name__}")

    databases.append(db)

n_matches = np.zeros((len(hash_methods), len(transformations)))

print("Computing top 5 accuracy...")
for index, image_file in tqdm(enumerate(image_files), total=len(image_files)):
    image = Image.open(os.path.join(dataset_folder, image_file)).convert("RGB")
    for j, transformation in enumerate(transformations):
        transformed_image = t.transform(image, transformation)
        for i, (hash_method, database) in enumerate(zip(hash_methods, databases)):
            modified_hash = hash_method([transformed_image])[0]
            result = database.query(modified_hash, k=5)
            if index in [point["index"] for point in result]:
                n_matches[i, j] += 1
    gc.collect()

for i, (hash_method, database) in enumerate(zip(hash_methods, databases)):
    for j, transformation in enumerate(transformations):
        print(f'{hash_method.__name__} with {transformation} transformation:', n_matches[i, j] / len(image_files))
    print("#############################################")