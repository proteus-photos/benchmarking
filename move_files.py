import os
import shutil
from tqdm import tqdm

source_folder = './adversarial_dataset'
dest_folder = './adversarial_data'
os.makedirs(os.path.join(dest_folder, "train", "clean"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "train", "adv"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "val", "clean"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "val", "adv"), exist_ok=True)

files = os.listdir(source_folder)
files.sort()
print(len(files), "files")


for i, filename in enumerate(tqdm(files)):
    if i<=int(0.8*len(files)):
        file_type = "train"
    else:
        file_type = "val"
    source_path = os.path.join(source_folder, filename)
    if filename.startswith("clean"):
        destination_path = os.path.join(dest_folder, file_type, "clean", filename.split("_")[1].replace(".jpg", ".png"))
    elif filename.startswith("adv"):
        destination_path = os.path.join(dest_folder, file_type, "adv", filename.split("_")[1].replace(".jpg", ".png"))
    else:
        print("whatttt", filename)
    shutil.move(source_path, destination_path)
