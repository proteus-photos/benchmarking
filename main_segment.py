import os
import numpy as np
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
from collections import defaultdict

from transformer import Transformer
from hashes.blockhash import blockhash
from hashes.neuralhash import neuralhash
from utils import match, create_bokehs, bbox_to_ltrb
from database import Database
from segment import Segmenter

from hashes.dhash import dhash
from hashes.ahash import ahash
from hashes.phash import phash
from hashes.whash import whash

def extract_hashes(image):
    hashed_bokehs = []

    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    segments = s.segment(image)
    bokehs = create_bokehs(image, blurred_image, [segment["segmentation"] for segment in segments])
    print(len(segments))
    for segment_index in range(len(segments)):
        left, top, right, bottom = bbox_to_ltrb(segments[segment_index]["bbox"])
        cropped_bokeh = bokehs[segment_index][top:bottom, left:right]
        cropped_bokeh = Image.fromarray(cropped_bokeh)

        hashed_bokeh = neuralhash([cropped_bokeh], array=True)[0]
        hashed_bokehs.append(hashed_bokeh)
    
    return hashed_bokehs, segments

#TODO: Implement np.packbits to reduce size of storage by 8x. Also test XOR on uint8 storage to see if speedup
BLUR_RADIUS = 20  # for 360x360
N_SEGMENT_RETRIEVAL = 10
N_IMAGE_RETRIEVAL = 5

open_image = lambda x: Image.open(x).convert("RGB").resize((360,360))

transformations = ['screenshot', 'crop', 'double screenshot'] #, 'jpeg']
hash_method = neuralhash

dataset_folder = './dataset/imagenet/images'
image_files = [f for f in os.listdir(dataset_folder)][:20]

t = Transformer()
s = Segmenter()

os.makedirs("databases", exist_ok=True)
databases = []

if hash_method.__name__ + "_segmented.npy" not in os.listdir("databases"):
    print("Creating database for", hash_method.__name__)
    mask_hashes = []
    image_numbers = []

    for i, image_file in tqdm(enumerate(image_files), total = len(image_files)):
        image = open_image(os.path.join(dataset_folder, image_file))

        hashed_bokehs, segments = extract_hashes(image)
        for hash in hashed_bokehs:
            mask_hashes.append(hash)
            image_numbers.append(i)

        gc.collect()

    db = Database(
        mask_hashes,
        storedir=f"segmented_databases/{hash_method.__name__}",
        metadata=image_numbers
    )

else:
    db = Database(
        None,
        storedir=f"segmented_databases/{hash_method.__name__}"
    )

n_matches = np.zeros(len(transformations))

print("Computing top 5 accuracy...")
for index, image_file in tqdm(enumerate(image_files), total=len(image_files)):
    image = open_image(os.path.join(dataset_folder, image_file))
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

    for j, transformation in enumerate(transformations):
        transformed_image = t.transform(image, transformation)

        hashed_bokehs, segments = extract_hashes(transformed_image)

        # holds the score for top mathched images in our database by sum(%age_match * area)
        image_score = defaultdict(float)
        for hashed_bokeh, segment in zip(hashed_bokehs, segments):
            results = db.query(hashed_bokeh, k=N_SEGMENT_RETRIEVAL)
            for result in results:
                image_score[result["metadata"]] += result["score"] * segment["area"]
        
        top_images = sorted(image_score.items(), key=lambda x: x[1], reverse=True)[:N_IMAGE_RETRIEVAL]
        top_image_indices = [image_number for image_number, _ in top_images]

        if index in top_image_indices:
            n_matches[j] += 1
    gc.collect()

for j, transformation in enumerate(transformations):
    print(f'{hash_method.__name__} with {transformation} transformation:', n_matches[j] / len(image_files))
print("#############################################")