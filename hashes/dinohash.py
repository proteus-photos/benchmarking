import torch
import numpy as np
from torchvision import transforms

def set_defense(defense_module, k=1):
    global defense, K
    K = k
    defense = defense_module

def set_differentiable(grad):
    global differentiable
    differentiable = grad

def load_model(path):
    global dinov2
    dinov2.load_state_dict(torch.load(path, weights_only=True))

model = "vits14_reg"
# Load model
dinov2 = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model}').cuda().eval()
for param in dinov2.parameters():
    param.requires_grad = False
dinov2.eval()

means = np.load(f'./hashes/dinov2_{model}_means.npy')
means_torch = torch.from_numpy(means).cuda().float()

components = np.load(f'./hashes/dinov2_{model}_PCA.npy').T
components_torch = torch.from_numpy(components).cuda().float()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

defense = None
K=1

def dinohash(image_arrays, differentiable=False, c=1, logits=False, l2_normalize=False, mydinov2=dinov2):
    # NOTE: differentiable assumes torch.Tensor input
    # NOTE: cpu is only supported for non-differentiable
    
    wrapper = torch.no_grad if not differentiable else torch.enable_grad
    if not isinstance(image_arrays, torch.Tensor):
        image_arrays = torch.stack([preprocess(im) for im in image_arrays])

    with wrapper():
        image_arrays = image_arrays.cuda()
        if defense is not None:
            for _ in range(K):
                image_arrays = defense.forward(image_arrays)

        image_arrays = normalize(image_arrays)
        
        outs = mydinov2(image_arrays) - means_torch
        
        outs = outs@components_torch

        if l2_normalize:
            outs = torch.nn.functional.normalize(outs, dim=1)
        outs *= c

        if not logits:
            if differentiable:
                outs = torch.sigmoid(outs)
            else:
                outs = outs >= 0
    
    del image_arrays
    
    return outs