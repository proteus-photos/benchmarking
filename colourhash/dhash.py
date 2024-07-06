import numpy as np
from PIL import Image
from tqdm import tqdm

def dhash(images, bits=100):
    hash_size = round(bits**0.5)
    hashes = []
    for image in tqdm(images):
        image = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(image.getdata(), dtype=int).reshape((hash_size + 1, hash_size))
        # compute differences
        diff = pixels[1:, :] > pixels[:-1, :]
        hashes.append(diff.flatten())
    return hashes