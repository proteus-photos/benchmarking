import sys
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import os
from tqdm import tqdm
from utils import chunk_call
import random
import torch.nn.functional as F
import io

class BaRTDefense:
    def __init__(self, min_transforms=2, max_transforms=5):
        self.min_transforms = min_transforms
        self.max_transforms = max_transforms
        
        # Define all possible transformations
        self.transformations = [
            self.jpeg_noise,
            # self.swirl,
            self.noise_injection,
            self.fft_perturbation,
            self.zoom,
            self.color_space,
            self.contrast,
            self.grayscale,
            self.denoise
        ]
        
        # Define color jitter transform
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        
        # Define gaussian blur for denoising
        self.gaussian_blur = transforms.GaussianBlur(
            kernel_size=(3, 3),
            sigma=(0.1, 2.0)
        )

    def apply_defense(self, image):
        """Apply random sequence of transformations to PIL image"""
        num_transforms = random.randint(self.min_transforms, self.max_transforms)
        selected_transforms = random.sample(self.transformations, num_transforms)
        
        img = image.copy()
        for transform in selected_transforms:
            img = transform(img)
            
        return img

    def jpeg_noise(self, img):
        """JPEG compression followed by noise"""
        # Apply JPEG compression
        buffer = io.BytesIO()
        quality = random.randint(50, 90)
        img.save(buffer, format='JPEG', quality=quality)
        img = Image.open(buffer)
        

        return Image.fromarray(img.astype(np.uint8))

    def swirl(self, img):
        """Swirl effect using grid sampling"""
        # Convert to tensor for grid sampling
        x = TF.to_tensor(img).unsqueeze(0)
        h, w = x.shape[-2:]
        
        # Create sampling grid
        theta = random.random() * 2 * np.pi
        strength = random.random() * 0.5
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        
        # Apply swirl transformation
        r = torch.sqrt(grid[..., 0]**2 + grid[..., 1]**2)
        angle = strength * torch.exp(-r)
        
        sin = torch.sin(angle)
        cos = torch.cos(angle)
        
        grid[..., 0] = cos * grid[..., 0] - sin * grid[..., 1]
        grid[..., 1] = sin * grid[..., 0] + cos * grid[..., 1]
        
        # Apply transformation and convert back to PIL
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')
        return TF.to_pil_image(x.squeeze(0))

    def noise_injection(self, img):
        """Multiple noise types"""
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        img_array = np.array(img)
        
        if noise_type == 'gaussian':
            std = random.uniform(0.01, 0.1)
            noise = np.random.normal(0, std * 255, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255)
        else:
            mask = np.random.random(img_array.shape[:2]) < 0.05
            noise = np.random.choice([0, 255], img_array.shape)
            noisy_img = np.where(mask[..., None], noise, img_array)
            
        return Image.fromarray(noisy_img.astype(np.uint8))

    def fft_perturbation(self, img):
        """FFT-based perturbation"""
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Apply FFT to each channel
        for c in range(3):
            freq = np.fft.fft2(img_array[..., c])
            h, w = freq.shape
            mask = np.ones_like(freq)
            
            # Modify high frequencies
            mask[:h//4, :] *= 0.9
            mask[-h//4:, :] *= 0.9
            mask[:, :w//4] *= 0.9
            mask[:, -w//4:] *= 0.9
            
            freq *= mask
            img_array[..., c] = np.real(np.fft.ifft2(freq))
            
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def zoom(self, img):
        """Random zoom and crop"""
        # Random zoom factor
        scale = 1 + random.random() * 0.5  # zoom 1x-1.5x
        
        # Calculate new dimensions
        w, h = img.size
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize and random crop
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = random.randint(0, new_w - w)
        top = random.randint(0, new_h - h)
        return img.crop((left, top, left + w, top + h))

    def color_space(self, img):
        """Color space transformations"""
        return self.color_jitter(img)

    def contrast(self, img):
        """Contrast adjustment"""
        factor = 0.7 + random.random() * 0.6  # range [0.7, 1.3]
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def grayscale(self, img):
        """Convert to grayscale"""
        return img.convert('L').convert('RGB')

    def denoise(self, img):
        """Denoising using gaussian blur"""
        return self.gaussian_blur(img)
    
# Load model
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').cuda().eval()
components = np.load(f"./hashes/dinoPCA.npy")
    
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

bart = BaRTDefense(min_transforms=2, max_transforms=6)

def dinohash(ims, bits=512, n_average=5, defense=False, *args, **kwargs):
    image_arrays = torch.stack([preprocess(im) for im in ims]).cuda()
    
    with torch.no_grad():
        outs = dinov2_vitb14_reg(image_arrays).cpu().numpy()
        outs = outs@components.T
        outs = outs >= 0

    return outs