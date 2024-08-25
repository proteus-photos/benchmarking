import numpy as np
from PIL import Image

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_keypoints
import torch
import torch.nn as nn
import torch.optim as optim

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

def create_model():
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 1280),
        nn.Hardswish(),
        # nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 4),
        nn.ReLU()
    )

    return model

def tilize(image, tiles):
    width, height = image.size
    tile_size = width // tiles
    
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image.crop((x, y, x+tile_size, y+tile_size))
            tiles.append(tile)
    return tiles