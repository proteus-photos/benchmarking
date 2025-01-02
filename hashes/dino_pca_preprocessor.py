import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(dataset_folder, self.image_files[idx])).convert("RGB"))
        return image


dataset_folder = './diffusion_data'
image_files = [f for f in os.listdir(dataset_folder)][:100_000]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 512

model = "dinov2_vits14_reg"
dataset = ImageDataset(image_files, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

dinov2 = torch.hub.load('facebookresearch/dinov2', model).cuda().eval()

outputs = []
for images in tqdm(dataloader):
    images = images.cuda()
    with torch.no_grad():
        output = dinov2(images).cpu()
    outputs.append(output)
        
outputs = torch.cat(outputs)
means = outputs.mean(dim=0, keepdim=True)
outputs -= means

pca = PCA(n_components=96, whiten=True)
pca.fit(outputs)

weights = pca.components_

np.save(f"./hashes/{model}_means", means.numpy())
np.save(f"./hashes/{model}_PCA", weights)