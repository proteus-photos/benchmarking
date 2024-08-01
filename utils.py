import numpy as np
from PIL import Image

X = 0
Y = 1
W = 2
H = 3

def match(original_hash, modified_hash):
    # difference = original_hash ^ modified_hash
    
    # mask = 0xFFFFFFFF
    # matching = ~(difference) & mask
    
    # matching_bits_count = bin(matching).count('1')
    return (original_hash == modified_hash).sum()

def create_bokehs(image, blurred, masks):
    image = np.array(image)
    blurred = np.array(blurred)
    
    bokeh_images = np.repeat(blurred[np.newaxis, ...], len(masks), axis=0)
    for i, mask in enumerate(masks):
        bokeh_images[i][mask] = image[mask]
    return bokeh_images

def bbox_to_ltrb(bbox):
    x, y, w, h = bbox
    return x, y, x+w, y+h

def clip_to_image(box, width, height):
    x = max(0, box[X])
    y = max(0, box[Y])
    w = min(width, box[X]+box[W]) - x
    h = min(height, box[Y]+box[H]) - y

    return [round(x), round(y), round(w), round(h)]