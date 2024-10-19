import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from utils import chunk_call
# Load model
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda().eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def dinohash(ims, bits=128, *args, **kwargs):
    global session
    # Load output hash matrix
    image_arrays = torch.stack([preprocess(im) for im in ims]).cuda()
    with torch.no_grad():
        outs = dinov2_vitb14_reg(image_arrays).cpu().numpy()

    hash_output = outs[:bits] > 0
    return hash_output