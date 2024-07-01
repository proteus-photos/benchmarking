import numpy as np
from PIL import Image

def dhash(images, bits=100):
    hash_size = round(bits**0.5)
    hashes = []
    for image in images:
        image = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = np.array(image.getdata(), dtype=int).reshape((hash_size + 1, hash_size))
        # compute differences
        diff = (pixels[1:, :] > pixels[:-1, :]).astype(int)
        hashes.append("".join(map(str, diff.flatten())))
    return hashes