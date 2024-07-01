import numpy as np
import pywt
from PIL import Image

def whash(images, bits=64, image_scale = None, mode = 'haar', remove_max_haar_ll = True):
    hash_size = round(bits**0.5)
    hashes = []
    for image in images:
        if image_scale is not None:
            assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
        else:
            image_scale = 2**int(np.log2(min(image.size)))
        ll_max_level = int(np.log2(image_scale))

        level = int(np.log2(hash_size))
        assert hash_size & (hash_size-1) == 0, "hash_size is not power of 2"
        assert level <= ll_max_level, "hash_size in a wrong range"
        dwt_level = ll_max_level - level

        image = image.convert("L").resize((image_scale, image_scale), Image.LANCZOS)
        pixels = np.array(image.getdata(), dtype=np.float16).reshape((image_scale, image_scale))
        pixels /= 255

        # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
        if remove_max_haar_ll:
            coeffs = pywt.wavedec2(pixels, 'haar', level = ll_max_level)
            coeffs = list(coeffs)
            coeffs[0] *= 0
            pixels = pywt.waverec2(coeffs, 'haar')

        # Use LL(K) as freq, where K is log2(@hash_size)
        coeffs = pywt.wavedec2(pixels, mode, level = dwt_level)
        dwt_low = coeffs[0]

        # Substract median and compute hash
        med = np.median(dwt_low)
        diff = (dwt_low > med).astype(int)
        hashes.append("".join(map(str, diff.flatten())))
    return hashes