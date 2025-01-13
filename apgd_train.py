import argparse
import os

from PIL import Image
import torch
from torch.optim import AdamW
from tqdm import tqdm

import numpy as np
from hashes.dinohash import preprocess, normalize, dinov2, dinohash
import torch
from apgd_attack import APGDAttack, criterion_loss

torch.manual_seed(0)
np.random.seed(0)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.hashes = []
        batches = [image_files[i:i+args.batch_size] for i in range(0, len(image_files), args.batch_size)]
        for batch in tqdm(batches):
            hashes = dinohash([Image.open(image_file) for image_file in batch], differentiable=False)
            self.hashes.append(hashes)
        self.hashes = torch.cat(self.hashes).float()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = preprocess(Image.open(image_file))
        return image, self.hashes[idx]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./dataset', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=50,
                    help='number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=2,
                    help='number of epochs')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)

image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))][:1024]

dataset = ImageDataset(image_files)
complete_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

apgd = APGDAttack(eps=args.epsilon)
optimizer = AdamW(dinov2.parameters(), lr=2e-5)

for epoch in range(args.n_epochs):

    for image_batch, hash_batch in tqdm(train_loader):
        hash_batch = hash_batch.cuda()

        adv_images, _ = apgd.attack_single_run(image_batch, hash_batch, args.n_iter, log=True)
        # adv_images = image_batch.cuda()

        optimizer.zero_grad()

        dinov2.train()

        adv_hash_batch, loss = criterion_loss(adv_images, hash_batch, loss="bce")
        loss = loss.mean()

        accuracy = (adv_hash_batch - hash_batch).abs().mean()
        print("Attack strength:", accuracy.item())

        loss.backward()
        optimizer.step()
        del hash_batch