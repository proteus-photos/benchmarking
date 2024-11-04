import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import gc
import argparse
import copy
import random
import cv2
import numpy as np
from collections import defaultdict

from transformer import Transformer
from utils import match, tilize, create_model, transform, reparametricize, chunk_call

from imwatermark import WatermarkEncoder, WatermarkDecoder

transformations = ["blur", "median", "brightness", "contrast"]

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:1_00]


t = Transformer()

encoder = WatermarkEncoder()
decoder = WatermarkDecoder('bits', 96)
hash = random.choices("01", k=96)
hash_arr = np.array([int(x) for x in hash])
encoder.set_watermark(wmType="bits", content=hash)

matches = defaultdict(list)

for image_file in tqdm(image_files):
    image_cv2 = cv2.imread(os.path.join(dataset_folder, image_file))
    encoded_image = encoder.encode(image_cv2, 'dwtDct')
    encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(encoded_image)

    for transformation in transformations:
        transformed_image = t.transform(t.transform(image_pil, "jpeg"), transformation)
        transformed_image = cv2.cvtColor(np.array(transformed_image), cv2.COLOR_RGB2BGR)

        mean_colour = transformed_image.mean(axis=0).mean(axis=0)
        crop_w = int(0.2 * transformed_image.shape[1])
        crop_h = int(0.2 * transformed_image.shape[0])

        # we pad the borders with the mean colour instead of cropping, so that image is not displaced
        # transformed_image[:crop_h] = mean_colour
        # transformed_image[-crop_h:] = mean_colour
        # transformed_image[:, :crop_w] = mean_colour
        # transformed_image[:, -crop_w:] = mean_colour
        
        watermark = decoder.decode(transformed_image, 'dwtDct')

        match = (watermark==hash_arr).mean()
        matches[transformation].append(match)

    del image_cv2, encoded_image, image_pil

for transformation, matches in matches.items():
    print(f"{transformation}: {np.mean(matches)} / {np.std(matches)}")