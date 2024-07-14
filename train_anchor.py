import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from transformer import Transformer
import os
from tqdm import tqdm

def rand_uniform(r1, r2):
    return torch.FloatTensor([1]).uniform_(r1, r2).item()

LEFT     = 0
TOP      = 1
RIGHT    = 2
BOTTOM   = 3
COMPRESS = 4

def box_loss(out1s, state1s, out2s, state2s):
    out1_xs, out1_ys = out1s[:, 0], out2s[:, 1]
    out2_xs, out2_ys = out2s[:, 0], out2s[:, 1]

    x1s = state1s[:, LEFT] + out1_xs * (state1s[:, RIGHT] - state1s[:, BOTTOM])
    y1s = state1s[:, TOP] + out1_ys * (state1s[:, BOTTOM] - state1s[:, TOP])

    x2s = state2s[:, LEFT] + out2_xs * (state2s[:, RIGHT] - state2s[:, BOTTOM])
    y2s = state2s[:, TOP] + out2_ys * (state2s[:, BOTTOM] - state2s[:, TOP])

    return torch.square(x1s-x2s).sum() + torch.square(y1s-y2s).sum()

t = Transformer()
class CustomDataset(Dataset):
    def __init__(self, image_directory, transform=None):
        self.image_paths = [os.path.join(image_directory, x) for x in os.listdir(image_directory)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def get_random_transform(self):
        MAX_CROP = 0.2

        compress = bool(torch.bernoulli(torch.Tensor([0.3])).item())  # 30% chance of compressing

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

        return image1, state1, image2, state2

if __name__ == "__main__":
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    for name, param in model.named_parameters():
        print(name)
        if "classifier" not in name and "features.12" not in name and "features.11" not in name and "features.10" not in name:
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 2),
        # nn.Hardswish(),
        # nn.Dropout(p=0.2, inplace=True),
        # nn.Linear(1024, 1)  # Output a single scalar
    )

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Create dataset and dataloader (you'll need to provide your own data)
    dataset = CustomDataset("./dataset/imagenet/images", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Training loop
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        for image1s, state1s, image2s, state2s in tqdm(train_loader):
            image1s = image1s.to(device)
            state1s = torch.stack(state1s, dim=1).to(device)

            image2s = image2s.to(device)
            state2s = torch.stack(state2s, dim=1).to(device)
            
            optimizer.zero_grad()

            output1s = model(image1s)
            output2s = model(image2s)

            loss = box_loss(output1s, state1s, output2s, state2s)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image1s, state1s, image2s, state2s in val_loader:
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")        

    print("Finetuning completed!")

    # Save the model
    torch.save(model.state_dict(), 'finetuned_mobilenetv3.pth')