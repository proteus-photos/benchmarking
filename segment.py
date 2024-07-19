from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.7]])
        img[m] = color_mask
    ax.imshow(img)

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

images = []
for path in os.listdir('images_test'):
    image = cv2.imread('images_test/'+path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=mobile_sam,
    points_per_side=16,
    points_per_batch=256,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.7,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

for i, image in enumerate(images):
    masks2 = mask_generator_2.generate(image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show()
    plt.savefig("wow.png")
    plt.close()