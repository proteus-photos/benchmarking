import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision

from PIL import Image
from transformer import Transformer
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

t = Transformer()
torch.manual_seed(42)

def rand_uniform(r1, r2):
    return torch.FloatTensor([1]).uniform_(r1, r2).item()

LEFT     = 0
TOP      = 1
RIGHT    = 2
BOTTOM   = 3
COMPRESS = 4

RED = 0
GREEN = 1
BLUE = 1

X = 0
Y = 1

BATCH_SIZE = 128

def original_coordinates(outs, states):
    xs, ys = outs[:, X], outs[:, Y]

    x1s = states[:, LEFT] + (xs/2 + .5) * (states[:, RIGHT] - states[:, BOTTOM])
    y1s = states[:, TOP] + (ys/2 + .5) * (states[:, BOTTOM] - states[:, TOP])

    return x1s, y1s

def box_loss(out1s, state1s, out2s, state2s):
    x1s, y1s = original_coordinates(out1s, state1s)
    x2s, y2s = original_coordinates(out2s, state2s)

    return (torch.square(x1s-x2s).sum() + torch.square(y1s-y2s).sum()) / len(out1s)

class CustomDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_paths = [os.path.join(image_directory, x) for x in os.listdir(image_directory)][:20]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def get_random_transform(self):
        MAX_CROP = 0.2

        compress = False # bool(torch.bernoulli(torch.Tensor([0.3])).item())  # 30% chance of compressing

        left = rand_uniform(0, MAX_CROP)
        top = rand_uniform(0, MAX_CROP)
        right = 1 - rand_uniform(0, MAX_CROP)
        bottom = 1 - rand_uniform(0, MAX_CROP)

        return (left, top, right, bottom, compress)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        state1 = self.get_random_transform()
        image1 = t.transform(image, 'crop', left=state1[LEFT], top=state1[TOP], right=state1[RIGHT], bottom=state1[BOTTOM])
        if state1[COMPRESS]:
            image1 = t.transform(image1, 'jpeg')
        image1 = self.transform(image1)

        state2 = self.get_random_transform()
        image2 = t.transform(image, 'crop', left=state2[LEFT], top=state2[TOP], right=state2[RIGHT], bottom=state2[BOTTOM])
        if state2[COMPRESS]:
            image2 = t.transform(image2, 'jpeg')
        image2 = self.transform(image2)

        return self.transform(image), image1, state1, image2, state2

if __name__ == "__main__":
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights)

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 1280),
        nn.Hardswish(),
        # nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 2),
        nn.Tanh()
    )

    classifier_params = []
    last_blocks_params = []

    for name, param in model.named_parameters():
        print(param.shape, name)
        # param.requires_grad = True

        if "classifier" in name:
            classifier_params.append(param)
            param.requires_grad = True
        # elif "features.16" in name or "features.15" in name or "features.14" in name:
        #     last_blocks_params.append(param)
        #     param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': classifier_params, 'lr': 1e-6},
    #     {'params': last_blocks_params, 'lr': 1e-6},
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
    ])


    dataset = CustomDataset("./dataset/imagenet/images", transform=transform)

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    # Training loop
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        for images, image1s, state1s, image2s, state2s in tqdm(train_loader):
            image1s = image1s.to(device)
            state1s = torch.stack(state1s, dim=1).to(device)

            image2s = image2s.to(device)
            state2s = torch.stack(state2s, dim=1).to(device)
            
            optimizer.zero_grad()

            output1s = model(image1s)
            output2s = model(image2s)

            # print(torch.log10(output1s.abs()).mean().item(), torch.log10(output2s.abs()).mean().item())
            # plt.scatter(output1s[:, X].detach().cpu().numpy(), output1s[:, Y].detach().cpu().numpy(), alpha=0.1, color='red', edgecolors=None)
            # plt.scatter(output2s[:, X].detach().cpu().numpy(), output2s[:, Y].detach().cpu().numpy(), alpha=0.1, color='blue', edgecolors=None)
            # plt.savefig(f"plots/{epoch+1}.png")
            # plt.close()

            xs, ys = original_coordinates(output1s, state1s)
            coordinates1 = torch.round(xs * 224).long(), torch.round(ys * 224).long()

            xs, ys = original_coordinates(output2s, state2s)
            coordinates2 = torch.round(xs * 224).long(), torch.round(ys * 224).long()
            
            # for coord1, coord2 in zip(zip(*coordinates1), zip(*coordinates2)):
            #     print(coord1[X].item(), coord1[Y].item(), coord2[X].item(), coord2[Y].item())

            images = inverse_transform(images)
            for i in range(len(images)):
                images[i, BLUE,  coordinates1[X][i]-3:coordinates1[X][i]+3, coordinates1[Y][i]-3:coordinates1[Y][i]+3] = 1
                images[i, GREEN, coordinates2[X][i]-3:coordinates2[X][i]+3, coordinates2[Y][i]-3:coordinates2[Y][i]+3] = 1

            grid_img = torchvision.utils.make_grid(images, nrow=5)
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.savefig(f"plots/image{epoch+1}.png")
            plt.close()

            loss = box_loss(output1s, state1s, output2s, state2s)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, image1s, state1s, image2s, state2s in val_loader:
                image1s = image1s.to(device)
                state1s = torch.stack(state1s, dim=1).to(device)

                image2s = image2s.to(device)
                state2s = torch.stack(state2s, dim=1).to(device)
                
                optimizer.zero_grad()

                output1s = model(image1s)
                output2s = model(image2s)

                loss = box_loss(output1s, state1s, output2s, state2s)            
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")        

    print("Finetuning completed!")

    # Save the model
    torch.save(model.state_dict(), 'finetuned_mobilenetv3.pth')