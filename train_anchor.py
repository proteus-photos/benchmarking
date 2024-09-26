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
from torchvision.utils import draw_bounding_boxes, draw_keypoints

from PIL import Image
from transformer import Transformer
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import argparse
import sys
from utils import transform, inverse_normalize, reparametricize, create_model, random_transform, normalize
import wandb

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

t = Transformer()

def rand_uniform(r1, r2):
    return torch.FloatTensor([1]).uniform_(r1, r2).item()

LEFT     = 0
TOP      = 1
RIGHT    = 2
BOTTOM   = 3
COMPRESS = 4

RED = 0
GREEN = 1
BLUE = 2

RED_COLOUR = torch.Tensor([1, 0, 0]).reshape(3, 1, 1)
GREEN_COLOUR = torch.Tensor([0, 1, 0]).reshape(3, 1, 1)
BLUE_COLOUR = torch.Tensor([0, 0, 1]).reshape(3, 1, 1)

X = 0
Y = 1
X1 = 0
Y1 = 1
X2 = 2
Y2 = 3

BATCH_SIZE = 128
IM_SIZE = 224

def myplot(images, output1s, state1s, output2s, state2s, epoch):
    images = inverse_normalize(images)
    x1s, y1s, x2s, y2s = original_coordinates(output1s, state1s)
    im1coordinates1 = torch.round(x1s * IM_SIZE).long(), torch.round(y1s * IM_SIZE).long()
    im1coordinates2 = torch.round(x2s * IM_SIZE).long(), torch.round(y2s * IM_SIZE).long()

    x1s, y1s, x2s, y2s = original_coordinates(output2s, state2s)
    im2coordinates1 = torch.round(x1s * IM_SIZE).long(), torch.round(y1s * IM_SIZE).long()
    im2coordinates2 = torch.round(x2s * IM_SIZE).long(), torch.round(y2s * IM_SIZE).long()
    
    lefts1 = torch.round(state1s[:, LEFT] * IM_SIZE).long()
    tops1 = torch.round(state1s[:, TOP] * IM_SIZE).long()
    rights1 = torch.round(state1s[:, RIGHT] * IM_SIZE).long()
    bottoms1 = torch.round(state1s[:, BOTTOM] * IM_SIZE).long()

    lefts2 = torch.round(state2s[:, LEFT] * IM_SIZE).long()
    tops2 = torch.round(state2s[:, TOP] * IM_SIZE).long()
    rights2 = torch.round(state2s[:, RIGHT] * IM_SIZE).long()
    bottoms2 = torch.round(state2s[:, BOTTOM] * IM_SIZE).long()

    for i in range(len(images)):

        # create bounding boxes
        images[i] = draw_bounding_boxes(images[i], torch.Tensor((lefts1[i], tops1[i], rights1[i], bottoms1[i])).unsqueeze(0), colors="blue", width=5)
        images[i] = draw_bounding_boxes(images[i], torch.Tensor((lefts2[i], tops2[i], rights2[i], bottoms2[i])).unsqueeze(0), colors="red", width=5)

        # place dots
        images[i] = draw_keypoints(images[i],
                                    torch.Tensor(((im1coordinates1[X][i], im1coordinates1[Y][i]),
                                                  (im1coordinates2[X][i], im1coordinates2[Y][i]))).unsqueeze(0),
                                    colors="blue", radius=5)
        images[i] = draw_keypoints(images[i],
                                    torch.Tensor(((im2coordinates1[X][i], im2coordinates1[Y][i]),
                                                  (im2coordinates2[X][i], im2coordinates2[Y][i]))).unsqueeze(0),
                                    colors="red", radius=5)
        
        images[i] = draw_keypoints(images[i],
                                    torch.Tensor(((im1coordinates1[X][i], im1coordinates1[Y][i]),
                                                  (im2coordinates1[X][i], im2coordinates1[Y][i]))).unsqueeze(0),
                                    colors="black", radius=2)
        images[i] = draw_keypoints(images[i],
                                    torch.Tensor(((im1coordinates2[X][i], im1coordinates2[Y][i]),
                                                  (im2coordinates2[X][i], im2coordinates2[Y][i]))).unsqueeze(0),
                                    colors="white", radius=2)
    grid_img = torchvision.utils.make_grid(images, nrow=5)
    im = Image.fromarray((grid_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    os.makedirs("plots", exist_ok=True)
    im.save(f"plots/image{epoch+1}.png")
    im.close()

class CustomDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_paths = [os.path.join(image_directory, x) for x in os.listdir(image_directory)[:1000]]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    # MAX CROP SHOULD BE 0.3 AND COMPRESS SHOULD BE 0.5!!!
    def get_random_transform(self, max_crop=0.2):

        compress = rand_uniform(0, 1) < 0.1  # 50% chance of compressing

        left = rand_uniform(0, max_crop)
        top = rand_uniform(0, max_crop)
        right = 1 - rand_uniform(0, max_crop)
        bottom = 1 - rand_uniform(0, max_crop)

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
    
def original_coordinates(outs, states):
    x1s, y1s, x2s, y2s = outs[:, X1], outs[:, Y1], outs[:, X2], outs[:, Y2]

    x1s = states[:, LEFT] + x1s * (states[:, RIGHT] - states[:, LEFT])
    y1s = states[:, TOP] + y1s * (states[:, BOTTOM] - states[:, TOP])

    x2s = states[:, LEFT] + x2s * (states[:, RIGHT] - states[:, LEFT])
    y2s = states[:, TOP] + y2s * (states[:, BOTTOM] - states[:, TOP])

    return x1s, y1s, x2s, y2s

def distance(point1, point2):
    # SHARPNESS = 3
    return torch.exp(-torch.abs(point1 - point2).sum(dim=1) * SHARPNESS) / SHARPNESS

def transform_point(point, anchors1, anchors2):
    # The resulting coordinates will be in the space of anchors2
    x11, y11, x12, y12 = anchors1.T
    x21, y21, x22, y22 = anchors2.T

    a_x = (x21 - x22) / (x11 - x12)
    b_x = x21 - a_x * x11
    
    a_y = (y21 - y22) / (y11 - y12)
    b_y = y21 - a_y * y11

    return a_x * point[X] + b_x, a_y * point[Y] + b_y

def box_loss(out1s, state1s, out2s, state2s):
    im1x1s, im1y1s, im1x2s, im1y2s = original_coordinates(out1s, state1s)
    im2x1s, im2y1s, im2x2s, im2y2s = original_coordinates(out2s, state2s)
    
    loss = torch.square(im1x1s-im2x1s).mean() + torch.square(im1y1s-im2y1s).mean() + torch.square(im1x2s-im2x2s).mean() + torch.square(im1y2s-im2y2s).mean()
    nearness_loss = distance(torch.stack((im1x1s, im1y1s), dim=1), torch.stack((im1x2s, im1y2s), dim=1)).mean() + \
                    distance(torch.stack((im2x1s, im2y1s), dim=1), torch.stack((im2x2s, im2y2s), dim=1)).mean()

    return loss * 30. + GAMMA*nearness_loss

def weighted_box_loss(out1s, state1s, out2s, state2s):
    # Multiplies box loss by IOU of the two crops
    losses = box_loss(out1s, state1s, out2s, state2s)

    left, _ = torch.max(torch.stack((state1s[:, LEFT], state2s[:, LEFT])), dim=0)
    top, _  = torch.max(torch.stack((state1s[:, TOP], state2s[:, TOP])), dim=0)
    right, _  = torch.min(torch.stack((state1s[:, RIGHT], state2s[:, RIGHT])), dim=0)
    bottom, _ = torch.min(torch.stack((state1s[:, BOTTOM], state2s[:, BOTTOM])), dim=0)

    intersection = (right - left) * (bottom - top)
    union = (state1s[:, RIGHT] - state1s[:, LEFT]) * (state1s[:, BOTTOM] - state1s[:, TOP]) + \
            (state2s[:, RIGHT] - state2s[:, LEFT]) * (state2s[:, BOTTOM] - state2s[:, TOP]) - \
            intersection
    ious = intersection / union
    
    return (losses * ious).mean()

def iou_loss(out1s, state1s, out2s, state2s):
    im1x1s, im1y1s, im1x2s, im1y2s = original_coordinates(out1s, state1s)
    im2x1s, im2y1s, im2x2s, im2y2s = original_coordinates(out2s, state2s)

    left, _ = torch.max(torch.stack((im1x1s, im2x1s), dim=1), dim=1)
    top, _ = torch.max(torch.stack((im1y1s, im2y1s), dim=1), dim=1)
    right, _ = torch.min(torch.stack((im1x2s, im2x2s), dim=1), dim=1)
    bottom, _ = torch.min(torch.stack((im1y2s, im2y2s), dim=1), dim=1)

    intersection = (right - left) * (bottom - top)
    union = (im1x2s - im1x1s) * (im1y2s - im1y1s) + (im2x2s - im2x1s) * (im2y2s - im2y1s) - intersection

    return 1 - (intersection / union).mean()

def overlap_loss(out1s, state1s, out2s, state2s):
    loss = 0

    # first calculate loss of 1st image transformed to 2nd image space
    left, top = transform_point((0,0), out1s, out2s)
    right, bottom = transform_point((1, 1), out1s, out2s)

    target_left = (state1s[:, LEFT] - state2s[:, LEFT]) / (state2s[:, RIGHT] - state2s[:, LEFT])
    target_top = (state1s[:, TOP] - state2s[:, TOP])/ (state2s[:, BOTTOM] - state2s[:, TOP])
    target_right = (state1s[:, RIGHT] - state2s[:, LEFT]) / (state2s[:, RIGHT] - state2s[:, LEFT])
    target_bottom = (state1s[:, BOTTOM] - state2s[:, TOP]) / (state2s[:, BOTTOM] - state2s[:, TOP])

    loss += torch.square(left - target_left).mean() + \
            torch.square(top - target_top).mean() +  \
            torch.square(right - target_right).mean() + \
            torch.square(bottom - target_bottom).mean()

    # then calculate loss of 2nd image transformed to 1st image space
    left, top = transform_point((0,0), out2s, out1s)
    right, bottom = transform_point((1, 1), out2s, out1s)

    target_left = (state2s[:, LEFT] - state1s[:, LEFT]) / (state1s[:, RIGHT] - state1s[:, LEFT])
    target_top = (state2s[:, TOP] - state1s[:, TOP])/ (state1s[:, BOTTOM] - state1s[:, TOP])
    target_right = (state2s[:, RIGHT] - state1s[:, LEFT]) / (state1s[:, RIGHT] - state1s[:, LEFT])
    target_bottom = (state2s[:, BOTTOM] - state1s[:, TOP]) / (state1s[:, BOTTOM] - state1s[:, TOP])

    loss += torch.square(left - target_left).mean() + \
            torch.square(top - target_top).mean() + \
            torch.square(right - target_right).mean() + \
            torch.square(bottom - target_bottom).mean()
    
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_coeff", type=float, default=4.0, help="Weight coefficient for the weighted box loss")
    parser.add_argument("--i_coeff", type=float, default=0.2, help="Weight coefficient for the IOU loss")
    parser.add_argument("--o_coeff", type=float, default=2.0, help="Weight coefficient for the overlap loss")
    parser.add_argument("--gamma", type=float, default=0.4, help="Gamma for the box loss function")
    parser.add_argument("--sharpness", type=float, default=5.0, help="Sharpness for the distance function")
    parser.add_argument("--lr1", type=float, default=4.5, help="Learning rate for the classifier")
    parser.add_argument("--lr2", type=float, default=4.0, help="Learning rate for the last blocks")
    parser.add_argument("--id", type=int, default=0, help="ID of the experiment")

    args = parser.parse_args()

    w_coeff = args.w_coeff
    i_coeff = args.i_coeff
    o_coeff = args.o_coeff

    lr1 = 10**(-args.lr1)
    lr2 = 10**(-args.lr2)
    
    GAMMA = args.gamma
    SHARPNESS = args.sharpness

    wandb.login()
    run = wandb.init(project=f"standard LR", config=args)

    model = create_model(checkpoint=None, backbone="mobilenet")

    # model.load_state_dict(torch.load('finetuned_mobilenetv3.pth'))

    classifier_params = []
    last_blocks_params = []
    for name, param in model.named_parameters():
        # print(name)
        # param.requires_grad = True

        # mobilenet                    resnet               resnet
        if "classifier" in name or "conv_block" in name or "fc" in name:
            classifier_params.append(param)
            param.requires_grad = True
        # mobilenet                                                    resnet
        elif any("features."+str(x) in name for x in range(9, 17)) or "fpn" in name:
            last_blocks_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam([
        {'params': classifier_params, 'lr': lr1},
        {'params': last_blocks_params, 'lr': lr2},
    ])


    dataset = CustomDataset("./dataset/imagenet/images", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    # Training loop
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    # if os.path.exists('checkpoint.pth'):
    #     checkpoint = torch.load('checkpoint.pth')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print(f"Checkpoint loaded. Starting from epoch {start_epoch+1}.")
    # else:

    start_epoch = 0
    #     print("No checkpoint found. Starting from scratch.")
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for step_no, (images, image1s, state1s, image2s, state2s) in enumerate(train_loader):
            image1s = normalize((image1s)).to(device)
            state1s = torch.stack(state1s, dim=1).to(device)

            image2s = normalize((image2s)).to(device)
            state2s = torch.stack(state2s, dim=1).to(device)
            
            optimizer.zero_grad()

            output1s = model(image1s)
            output1s = reparametricize(output1s)
            output2s = model(image2s)
            output2s = reparametricize(output2s)
            
            loss = w_coeff*weighted_box_loss(output1s, state1s, output2s, state2s) + o_coeff*overlap_loss(output1s, state1s, output2s, state2s) + i_coeff*iou_loss(output1s, state1s, output2s, state2s)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            train_loss += loss.item()
            if (step_no % 80 == 0):
                print(f"Step [{step_no+1}/{len(train_loader)}], Train Loss: {train_loss/(step_no+1):.6f}")
            
        train_loss /= len(train_loader)
        myplot(normalize(random_transform(images)), output1s, state1s, output2s, state2s, args.id)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, image1s, state1s, image2s, state2s in val_loader:
                image1s = normalize(image1s).to(device)
                state1s = torch.stack(state1s, dim=1).to(device)

                image2s = normalize(image2s).to(device)
                state2s = torch.stack(state2s, dim=1).to(device)
                
                optimizer.zero_grad()

                output1s = model(image1s)
                output1s = reparametricize(output1s)
                output2s = model(image2s)
                output2s = reparametricize(output2s)

                loss = overlap_loss(output1s, state1s, output2s, state2s)            
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Save checkpoint
        # torch.save({
        #     'epoch': epoch+1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        # }, 'checkpoint.pth')
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        # myplot(normalize(images), output1s, state1s, output2s, state2s, args.id)

    print(f"RETURN_VALUE:{val_loss}", file=sys.stderr)

    # Save the model
    # torch.save(model.state_dict(), 'finetuned_mobilenetv3.pth')