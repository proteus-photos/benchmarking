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
import sys
from tqdm import tqdm

X = 0
Y = 1
W = 2
H = 3

X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

def reparametricize(outs, MIN_MARGIN):
    # converts two floats (0, inf) to left margin and right margin
    # s.t. second point always >= first point

    # MIN_MARGIN = 0.3  # actual minimum distance = MIN_MARGIN / (2 + MIN_MARGIN)

    x1s, y1s, x2s, y2s = outs[:, X1], outs[:, Y1], outs[:, X2], outs[:, Y2]
    normalized = (
        x1s / (x1s + x2s + MIN_MARGIN),
        y1s / (y1s + y2s + MIN_MARGIN), 
        (x1s + MIN_MARGIN) / (x1s + x2s + MIN_MARGIN),
        (y1s + MIN_MARGIN) / (y1s + y2s + MIN_MARGIN)
    )

    if torch.all(normalized[0] < 1e-5) and torch.all(torch.all(normalized[1] < 1e-5)) and torch.all(normalized[2] > 1-1e-5) and torch.all(normalized[3] > 1-1e-5):
        print(f"RETURN_VALUE:{-1}", file=sys.stderr)
        exit()
        
    return torch.stack(normalized, dim=1)

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

def create_model(checkpoint=None):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 1280),
        nn.Hardswish(),
        # nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 4),
        nn.ReLU()
    )

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))

    return model.cuda()

def tilize(image, n_tiles):
    width, height = image.size
    tile_size = width / n_tiles
    
    tiles = []
    for y in np.linspace(0., float(height), n_tiles, endpoint=False):
        for x in np.linspace(0, float(width), n_tiles, endpoint=False):
            tile = image.crop((x, y, x+tile_size, y+tile_size))
            tiles.append(tile)
    return tiles

@torch.no_grad
def chunk_call(model, inputs, batchsize=256):
    outputs = []
    for i in tqdm(range(0, len(inputs), batchsize)):
        outputs.append(model(inputs[i:i+batchsize].cuda()).cpu())
    return torch.cat(outputs)