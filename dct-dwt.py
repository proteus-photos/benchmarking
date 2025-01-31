import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import cv2
import numpy as np
from collections import defaultdict
import pandas as pd
from scipy.stats import binom

from transformer import Transformer
from utils import match

from imwatermark import WatermarkEncoder, WatermarkDecoder

transformations = ["blur", "median", "brightness", "contrast"]

dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:50_000]

BITS = 96

t = Transformer()

def combined_transform(image):
    transformations = ["jpeg", transformation, "erase", "text"]
    for transform in transformations:
        image = t.transform(image, method=transform)
    return image

def generate_roc(matches, bits):
    matches = matches * bits
    taus = np.arange(bits+1)
    tpr = [(matches>=tau).mean() for tau in taus]

    fpr = 1 - binom.cdf(taus-1, bits, 0.5)
    
    df = pd.DataFrame({
        "tpr": tpr,
        "fpr": fpr,
        "tau": taus
    })
    
    df.to_csv(f"./results/dctdwt_{transformation}.csv")

encoder = WatermarkEncoder()
decoder = WatermarkDecoder('bits', BITS)
hash = random.choices("01", k=BITS)
hash_arr = np.array([int(x) for x in hash])
encoder.set_watermark(wmType="bits", content=hash)

matches = defaultdict(list)

for image_file in tqdm(image_files):
    image_cv2 = cv2.imread(os.path.join(dataset_folder, image_file))
    if min(image_cv2.shape[:2]) <= 256:
        continue
    encoded_image = encoder.encode(image_cv2, 'dwtDct')
    encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(encoded_image)

    for transformation in transformations:
        transformed_image = combined_transform(image_pil)
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
    matches = np.array(matches)
    print(f"{transformation}: {np.mean(matches)} / {np.std(matches)}")

    generate_roc(matches, BITS)