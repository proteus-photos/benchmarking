import os
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import random
import io

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', **kwargs):
        self.repeat = repeat
        self.cache = cache
        self.train_or_val = kwargs.get('train_or_val', None)
        self.im_size = kwargs.get('im_size', None)
        self.jpeg = kwargs.get('jpeg', False)

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]

        if kwargs.get("max_len", None) is not None:
            filenames = filenames[:kwargs["max_len"]]

        if self.train_or_val == 'train':
            filenames = filenames[:int(len(filenames) * 0.8)]
        elif self.train_or_val == 'val':
            filenames = filenames[int(len(filenames) * 0.8):]

        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            im =  Image.open(x).convert('RGB')
            if self.im_size is not None:
                im = im.resize((self.im_size, self.im_size))
            # if we doing jpeg theres a 20% compression happening
            if self.jpeg and random.random() < 0.2:
                quality = random.randint(70, 100)

                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='JPEG', quality=quality)
                img_byte_arr = img_byte_arr.getvalue()
                
                im = Image.open(io.BytesIO(img_byte_arr))
            
            return transforms.ToTensor()(im)

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_adv = ImageFolder(root_path_1, jpeg=True, **kwargs)
        max_len = len(os.listdir(root_path_1))
        im_size = Image.open(os.path.join(root_path_1, os.listdir(root_path_1)[0])).size[0]

        self.dataset_clean = ImageFolder(root_path_2, max_len=max_len, im_size=im_size, **kwargs)

    def __len__(self):
        return len(self.dataset_adv)

    def __getitem__(self, idx):
        return self.dataset_adv[idx], self.dataset_clean[idx]
