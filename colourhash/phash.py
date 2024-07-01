import scipy.fftpack
import numpy as np
from PIL import Image

def phash(images, bits=100, highfreq_factor=4):
    hash_size = round(bits**0.5)

    hashes = []
    for image in images:
        img_size = hash_size * highfreq_factor
        image = image.convert("L").resize((img_size, img_size), Image.LANCZOS)
        pixels = np.array(image.getdata(), dtype=np.float16).reshape((img_size, img_size))
        dct = scipy.fftpack.dct(pixels)
        dctlowfreq = dct[:hash_size, 1:hash_size+1]
        avg = dctlowfreq.mean()
        diff = (dctlowfreq > avg).flatten()
        hashes.append("".join(map(str,diff.astype(int))))
	
    return hashes