import argparse
import os

from PIL import Image
import torch
from torch.optim import AdamW
from tqdm import tqdm
import copy

import numpy as np
from hashes.dinohash import preprocess, normalize, dinov2, dinohash
import torch
from apgd_attack import APGDAttack, criterion_loss
from utils import AverageMeter

torch.manual_seed(0)
np.random.seed(0)

BITS = 96

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files):
        self.image_files = image_files
        self.computed = torch.zeros(len(image_files), dtype=torch.bool)
        self.logits = torch.zeros(len(image_files), BITS)
        self.batch_size = 16

        self.mydinov2 = copy.deepcopy(dinov2)
        for param in self.mydinov2.parameters():
            param.requires_grad = False
        self.mydinov2.eval()

        # self.logits = []
        # batchSize = 4096
        # batches = [image_files[i:i+batchSize] for i in range(0, len(image_files), batchSize)]
        # for batch in tqdm(batches):
        #     logits = dinohash([Image.open(image_file) for image_file in batch], differentiable=False, logits=True, c=1).cpu()
        #     self.logits.append(logits)
        # self.logits = torch.cat(self.logits).float()
        # np.save('./logits.npy', self.logits.numpy())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if not self.computed[idx]:
            start = (idx // self.batch_size) * self.batch_size
            end = min(start + self.batch_size, len(self.image_files))
            images = [preprocess(Image.open(image_file)) for image_file in self.image_files[start:end]]
            images = torch.stack(images)
            logits = dinohash(images, differentiable=False, logits=True,
                              c=1, mydinov2=self.mydinov2)
            self.logits[start:end] = logits
            self.computed[start:end] = True
        image_file = self.image_files[idx]
        image = preprocess(Image.open(image_file))

        return image, self.logits[idx]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Perform neural collision attack.')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256,
                    help='batch size for processing images')
parser.add_argument('--image_dir', dest='image_dir', type=str,
                    default='./diffusion_data', help='directory containing images')
parser.add_argument('--n_iter', dest='n_iter', type=int, default=10,
                    help='average number of iterations')
parser.add_argument('--n_iter_range', dest='n_iter_range', type=int, default=0,
                    help='maximum number of iterations')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=8/255,
                    help='maximum perturbation (L∞ norm bound)')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=1,
                    help='number of epochs')
parser.add_argument('--lr', dest='lr', type=float, default=1e-6,
                    help='learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--warmup', dest='warmup', type=int, default=1_40,
                    help='number of warmup steps')
parser.add_argument('--steps', dest='steps', type=int, default=20_00,
                    help='number of steps')
parser.add_argument('--start_step', dest='start_step', type=int, default=0,
                    help='starting step')
parser.add_argument('--clean_weight', dest='clean_weight', type=float, default=10,
                    help='weight of clean loss')
parser.add_argument('--val_freq', dest='val_freq', type=int, default=50,
                    help='validation frequency')

args = parser.parse_args()
os.makedirs('./adversarial_dataset', exist_ok=True)

image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
image_files.sort()
image_files = image_files[:50_631]
# 1_052_631
dataset = ImageDataset(image_files)
complete_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

SPLIT_RATIO = 0.99
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(SPLIT_RATIO*len(dataset)), len(dataset)-int(SPLIT_RATIO*len(dataset))])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

apgd = APGDAttack(eps=args.epsilon)
optimizer = AdamW(dinov2.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)
step_total = args.start_step
epoch_total = 0

for param in dinov2.parameters():
    param.requires_grad = True

while step_total < args.steps:
    pbar = tqdm(train_loader)
    loss_meter = AverageMeter('Loss')
    accuracy_meter = AverageMeter('Accuracy')

    for images, logits in pbar:
        scheduler(step_total)
        n_iter = np.random.randint(args.n_iter - args.n_iter_range,
                                   args.n_iter + args.n_iter_range + 1)

        logits = logits.cuda()
        images = images.cuda()

        # logits = dinohash(images, differentiable=False, logits=True, c=1).float().cuda()

        adv_images, _ = apgd.attack_single_run(images, logits, n_iter, log=False)

        # adv_images = images.cuda()

        dinov2.train()
        adv_hashes, adv_loss = criterion_loss(adv_images, logits, loss="target bce", l2_normalize=False)

        adv_loss = adv_loss.mean()
        adv_loss.backward()

        clean_loss = 0
        if args.clean_weight > 0:
            clean_hashes, clean_loss = criterion_loss(images, logits, loss="target bce", l2_normalize=False)
            clean_loss =  args.clean_weight * clean_loss.mean()
            clean_loss.backward()

        loss = adv_loss + clean_loss
        adv_loss = adv_loss.item()
        clean_loss = clean_loss.item() / args.clean_weight

        optimizer.step()
        optimizer.zero_grad()

        hashes = (logits >= 0).float()
        accuracy = (adv_hashes - hashes).abs().mean()

        loss_meter.update(loss.item(), len(images))
        accuracy_meter.update(accuracy.item(), len(images))

        pbar.set_description(f"attack: {accuracy * 100:.4f}, loss: {loss:.4f}, adv_loss: {adv_loss:.4f}, clean_loss: {clean_loss:.4f}")

        step_total += 1

        del hashes, logits, images, adv_images, adv_hashes

        if step_total % args.val_freq == 0:
            dinov2.eval()

            total_strength = 0
            total_accuracy = 0
            n_images = 0

            for images, logits in test_loader:
                logits = logits.cuda()
                hashes = (logits >= 0).float()

                adv_images, _ = apgd.attack_single_run(images, logits, n_iter=20)

                adv_hashes = dinohash(adv_images).float()
                accuracy = (adv_hashes - hashes).cpu().abs().mean().item()

                clean_hashes = dinohash(images).float()
                clean_accuracy = (clean_hashes - hashes).cpu().abs().mean().item()

                total_strength += accuracy * len(images)
                total_accuracy += clean_accuracy * len(images)
                n_images += len(images)

                del hashes, logits, adv_images, adv_hashes, images, clean_hashes

            print(f"validation attack strength: {total_strength / n_images * 100:.2f}%, clean error:  {total_accuracy / n_images * 100:.2f}%")

    print(f"step: {step_total}, loss: {loss_meter.avg:.4f}, accuracy: {accuracy_meter.avg:.4f}")
    del loss_meter, accuracy_meter