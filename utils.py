import numpy as np
from PIL import Image

from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as transforms

import torchvision
from torchvision.utils import draw_bounding_boxes, draw_keypoints
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm

class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        
        fasterrcnn = fasterrcnn_resnet50_fpn_v2(pretrained=True)
        
        self.backbone = fasterrcnn.backbone
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 4)
        
    def forward(self, x):
        features = self.backbone(x)
                
        x = self.conv_block(features['3'])
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        return x
    
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
])

random_transform = transforms.Compose([
    # transform,
    transforms.GaussianNoise(0., 0.05),
    # transforms.ElasticTransform(alpha=10., sigma=10.),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
])

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


inverse_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])


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

def create_model(checkpoint=None, backbone="mobilenet"):
    if backbone == "mobilenet":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 1280),
            nn.Hardswish(),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 6),
            # nn.ReLU()
        )
    elif backbone == "mobilenet_old":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 1280),
            nn.Hardswish(),
            # nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 4),
            nn.ReLU()
        )
    elif backbone == "resnet":
        model = ResNetModel()
    else:
        raise ValueError("Invalid backbone")

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

def tilize_by_anchors(image, n_breaks, anchors):
    width, height = image.size

    tile_width  = float((anchors[X2] - anchors[X1])/n_breaks)
    tile_height = float((anchors[Y2] - anchors[Y1])/n_breaks)


    # compute number of grid lines in left and right directions
    n_max_pos_width = (1 - anchors[X1] + 1e-6) // tile_width
    n_max_neg_width =  - ((anchors[X1] + 1e-6) // tile_width)
    n_values = np.arange(n_max_neg_width, n_max_pos_width + 1)
    # Compute the corresponding values of x + n * dx
    width_values = (anchors[X1] + n_values * tile_width) * width

    # compute number of grid lines in up and down directions
    n_max_pos_height = (1 - anchors[Y1] + 1e-6) // tile_height
    n_max_neg_height = - ((anchors[Y1] + 1e-6) // tile_height)
    n_values = np.arange(n_max_neg_height, n_max_pos_height + 1)

    # Compute the corresponding values of y + n * dy
    height_values = (anchors[Y1] + n_values * tile_height) * height
    
    tile_width *= width
    tile_height *= height

    tiles = [image.crop((x, y, x+tile_width, y+tile_height)) for y in height_values[:-1] for x in width_values[:-1]]
    n_range = (n_max_neg_width, n_max_neg_height, n_max_pos_width, n_max_pos_height) #ltrb

    return tiles, [int(num) for num in n_range]

@torch.no_grad
def chunk_call(model, inputs, batchsize=128):
    if model is None:
        return torch.zeros(inputs.size(0), 4)
    outputs = []
    for i in tqdm(range(0, len(inputs), batchsize)):
        outputs.append(model(inputs[i:i+batchsize].cuda()).cpu())
    return torch.cat(outputs)