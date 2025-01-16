import torch
import numpy as np
from torchvision import transforms

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

def dinohash(image_arrays, differentiable=False, c=5, logits=False, tensor=False, l2_normalize=True):
    # NOTE: differentiable assumes torch.Tensor input
    assert (not logits) or (differentiable), "logits only supported in differentiable mode"

    dinov2.eval()

    if not differentiable:
        if not tensor:
            image_arrays = torch.stack([preprocess(im) for im in image_arrays])
        
        image_arrays = normalize(image_arrays.cuda())
        with torch.no_grad():
            outs = dinov2(image_arrays) - means_torch
            outs = outs@components_torch
            outs = outs >= 0
    else:
        image_arrays = normalize(image_arrays)
        outs = dinov2(image_arrays) - means_torch
        outs = outs@components_torch
        if l2_normalize:
            outs = torch.nn.functional.normalize(outs, dim=1) * c
        if not logits:
            outs = torch.sigmoid(outs)
    
    del image_arrays
    return outs