import sys
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
import random
import torch.nn.functional as F
import io

model = "vits14_reg"
# Load model
dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model}').cuda().eval()

means = np.load(f'./hashes/dinov2_{model}_means.npy')
means_torch = torch.from_numpy(means).cuda().float()

components = np.load(f'./hashes/dinov2_{model}_PCA.npy').T
components_torch = torch.from_numpy(components).cuda().float()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def dinohash(ims, differentiable=False, c=5, logits=False):
    # NOTE: non-differentiable assumes PIL input and differentiable assumes torch.Tensor input
    assert (not logits) or (differentiable), "logits only supported in differentiable mode"

    if not differentiable:
        image_arrays = torch.stack([normalize(preprocess(im)) for im in ims]).cuda()
        with torch.no_grad():
            outs = dinov2(image_arrays).cpu().numpy() - means
            outs = outs@components
            outs = outs >= 0
    else:
        image_arrays = normalize(ims)
        outs = dinov2(image_arrays) - means_torch
        outs = outs@components_torch
        outs = torch.nn.functional.normalize(outs, dim=1)
        if not logits:
            outs = torch.sigmoid(outs * c)
        del image_arrays
    return outs