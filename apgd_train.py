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

        # self.logits = []
        # batchSize = 4096
        # batches = [image_files[i:i+batchSize] for i in range(0, len(image_files), batchSize)]
        # for batch in tqdm(batches):
        #     logits = dinohash([Image.open(image_file) for image_file in batch], differentiable=False, logits=True, c=1).cpu()
        #     self.logits.append(logits)
        # self.logits = torch.cat(self.logits).float()
        # np.save('./logits.npy', self.logits.numpy())

        self.logits = torch.from_numpy(np.load('./logits.npy'))[:len(image_files)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = preprocess(Image.open(image_file))
        return image, self.logits[idx]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./diffusion_data', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=10,
                    help='average number of iterations')
parser.add_argument('--n_iter_range', dest='n_iter_range', type=int, default=3,
                    help='maximum number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=1e-7,
                    help='learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                    help='weight decay')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)

image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
image_files.sort()
image_files = image_files[:10_000]

dataset = ImageDataset(image_files)
complete_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

SPLIT_RATIO = 0.9
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(SPLIT_RATIO*len(dataset)), len(dataset)-int(SPLIT_RATIO*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

apgd = APGDAttack(eps=args.epsilon)
optimizer = AdamW(dinov2.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.n_epochs):

    for image_batch, logits_batch in tqdm(train_loader):
        n_iter = np.random.randint(args.n_iter - args.n_iter_range,
                                   args.n_iter + args.n_iter_range + 1)

        logits_batch = logits_batch.cuda()
        image_batch = image_batch.cuda()

        # logits_batch = dinohash(image_batch, differentiable=False, logits=True, c=1).float().cuda()
        hash_batch = (logits_batch >= 0).float()

        adv_images, _ = apgd.attack_single_run(image_batch, logits_batch, n_iter, log=False)

        # adv_images = image_batch.cuda()

        optimizer.zero_grad()

        dinov2.train()
        adv_hash_batch, loss = criterion_loss(adv_images, logits_batch, loss="target bce", l2_normalize=False)

        loss = loss.mean()

        accuracy = (adv_hash_batch - hash_batch).abs().mean()
        print("Attack strength:", accuracy.item())

        loss.backward()
        optimizer.step()
        del hash_batch, logits_batch, image_batch

    dinov2.eval()

    total_strength = 0
    n_images = 0

    for image_batch, logits_batch in tqdm(test_loader):
        logits_batch = logits_batch.cuda()
        hash_batch = (logits_batch >= 0).float()

        adv_images, _ = apgd.attack_single_run(image_batch, hash_batch, 10)

        adv_hash_batch = dinohash(adv_images, differentiable=False).float()
        accuracy = (adv_hash_batch - hash_batch).cpu().abs().mean().item()

        total_strength += accuracy * len(image_batch)
        n_images += len(image_batch)

        del hash_batch, logits_batch

    print("Validation attack strength:")
    print(total_strength / n_images)