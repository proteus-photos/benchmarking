import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from transformer import Transformer
from blockhash.blockhash import blockhash
from neuralhash.neuralhash import neuralhash

from colourhash.dhash import dhash
from colourhash.ahash import ahash
from colourhash.phash import phash
from colourhash.whash import whash

def match(original_hash, modified_hash):
    return sum(c1 == c2 for c1, c2 in zip(modified_hash, original_hash)) / len(modified_hash)

dataset_folder = './dataset'
image_files = [f for f in os.listdir(dataset_folder)]
images = [Image.open(os.path.join(dataset_folder, image_file)) for image_file in image_files]

transformations = ['jpeg', 'crop', 'screenshot', 'double screenshot']
hash_methods = [blockhash, neuralhash, dhash, ahash, phash, whash]

original_hashes_per_method = [hash_method(images) for hash_method in hash_methods]
bit_match_percentage = np.zeros((len(hash_methods), len(transformations), len(image_files)))
transformer = Transformer()

transformed_images_list = []
for transformation in transformations:
    transformed_images = []
    for image in images:
        transformed_image = transformer.transform(image, transformation)
        transformed_images.append(transformed_image)
    transformed_images_list.append(transformed_images)

for i, (original_hashes, hash_method) in enumerate(zip(original_hashes_per_method, hash_methods)):
    for j, transformed_images in tqdm(enumerate(transformed_images_list)):
        modified_hashes = hash_method(transformed_images)
        bit_overlap_percentages = [match(modified_hash, original_hash) for modified_hash, original_hash in zip(modified_hashes, original_hashes)]
        bit_match_percentage[i, j] = bit_overlap_percentages

for i, hash_method in enumerate(hash_methods):
    for j, transformation in enumerate(transformations):
        print(f'{hash_method.__name__} with {transformation} transformation:', np.mean(bit_match_percentage[i, j]))
        plt.hist(bit_match_percentage[i, j], bins=20)
        plt.savefig(f'distribution/{hash_method.__name__}_{transformation}.png')
        plt.close()

    print("#############################################")