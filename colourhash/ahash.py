import numpy as np
from PIL import Image

def ahash(images, bits=100):
    hash_size = round(bits**0.5)
    hashes = []
    for image in images:
        image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
        pixels = np.array(image.getdata()).reshape((hash_size, hash_size))
        avg = pixels.mean()
        diff = (pixels > avg).flatten()
        # make a hash
        hashes.append("".join(map(str, diff)))
    return hashes