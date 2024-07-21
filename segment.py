from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def combine(mask1, mask2):

    return {
            "segmentation": mask1["segmentation"] | mask2["segmentation"],
            "area": mask1["area"] + mask2["area"],
            "bbox": [
                min(mask1["bbox"][0], mask2["bbox"][0]),
                min(mask1["bbox"][1], mask2["bbox"][1]),
                max(mask1["bbox"][0] + mask1["bbox"][2], mask2["bbox"][0] + mask2["bbox"][2]),
                max(mask1["bbox"][1] + mask1["bbox"][3], mask2["bbox"][1] + mask2["bbox"][3])
            ],
            "predicted_iou": (mask1["predicted_iou"] * mask1["area"] + mask2["predicted_iou"] * mask2["area"]) / (mask1["area"] + mask2["area"]),
            "point_coords": [[
                (mask1["point_coords"][0][0] * mask1["area"] + mask2["point_coords"][0][0] * mask2["area"]) / (mask1["area"] + mask2["area"]),
                (mask1["point_coords"][0][1] * mask1["area"] + mask2["point_coords"][0][1] * mask2["area"]) / (mask1["area"] + mask2["area"])
            ]],
            "stability_score": (mask1["stability_score"] * mask1["area"] + mask2["stability_score"] * mask2["area"]) / (mask1["area"] + mask2["area"]),
            "crop_box": [
                min(mask1["crop_box"][0], mask2["crop_box"][0]),
                min(mask1["crop_box"][1], mask2["crop_box"][1]),
                max(mask1["crop_box"][0] + mask1["crop_box"][2], mask2["crop_box"][0] + mask2["crop_box"][2]),
                max(mask1["crop_box"][1] + mask1["crop_box"][3], mask2["crop_box"][1] + mask2["crop_box"][3]),
            ]
        }

def intersection_over_smaller(boxA, boxB):
    # xA = max(boxA["bbox"][0], boxB["bbox"][0])
    # yA = max(boxA["bbox"][1], boxB["bbox"][1])
    # xB = min(boxA["bbox"][0] + boxA["bbox"][2], boxB["bbox"][0] + boxB["bbox"][2])
    # yB = min(boxA["bbox"][1] + boxA["bbox"][3], boxB["bbox"][1] + boxB["bbox"][3])

    # interArea = max(0, xB - xA) * max(0, yB - yA)
    # boxAArea = boxA["bbox"][2] * boxA["bbox"][3]
    # boxBArea = boxB["bbox"][2] * boxB["bbox"][3]

    interArea = (boxA["segmentation"] & boxB["segmentation"]).sum()
    boxAArea = boxA["segmentation"].sum()
    boxBArea = boxB["segmentation"].sum()

    return interArea / min(boxAArea, boxBArea)

def supress_subsets(masks, n_final_masks=4):
    # combines masks based on high ios scores until there are n_final_masks left
    # returns a list of masks
    while len(masks) > n_final_masks:

        # NOTE: Diagonal will be zero (even though technicall it should be 1)
        # We do this so that the argmax doesnt get stuck at diagonals always

        ios_matrix = np.zeros((len(masks), len(masks)))
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                ios = intersection_over_smaller(masks[i], masks[j])
                ios_matrix[i, j] = ios
                ios_matrix[j, i] = ios
                
        mask1_index, mask2_index = np.unravel_index(np.argmax(ios_matrix, axis=None), ios_matrix.shape)
        # dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])

        # combine the two dictionaries based on weighted average of area store in mask1_index, delete mask2_index
        masks[mask1_index] = combine(masks[mask1_index], masks[mask2_index])
        ios_matrix[mask1_index] = [intersection_over_smaller(masks[mask1_index], masks[i]) if mask1_index!=i else 0 for i in range(len(masks))]
        ios_matrix[:, mask1_index] = ios_matrix[mask1_index]

        masks.pop(mask2_index)
        ios_matrix = np.delete(np.delete(ios_matrix, mask2_index, axis=0), mask2_index, axis=1)

    return masks

model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    mobile_sam,
    points_per_side=8,  # Reduced significantly to get larger segments
    min_mask_region_area=1024
)

class Segmenter:
    def __init__(self):
        pass

    def segment(self, image):
        masks = mask_generator.generate(np.array(image))
        if len(masks) == 0:
            return [{
                'segmentation': np.ones((image.size[1], image.size[0]), dtype=bool),
                'area': image.size[1]*image.size[0],
                'bbox': [0, 0, image.size[1], image.size[0]],
                'predicted_iou': -1,
                'point_coords': [image.size[1]//2, image.size[0]//2],
                'stability_score': -1,
                'crop_box': [0, 0, image.size[1], image.size[0]]
            }]
        return supress_subsets(masks, n_final_masks=4)
    
if __name__ == "__main__":
    s = Segmenter()
    images = []
    for path in os.listdir('images_test'):
        image = Image.open(f'images_test/{path}').convert("RGB")
        image = np.array(image)
        images.append(image)

    for i, image in tqdm(enumerate(images)):
        masks = s.segment(image)
        print(len(masks), [mask['area'] for mask in masks])
        
        if len(masks) == 0:
            continue
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        
        for j, ann in enumerate(sorted_anns):
            segmented_img = image.copy()
            m = ann['segmentation']
            segmented_img[m] = np.array([0, 0, 255])
            save_img = Image.fromarray(segmented_img)
            save_img.save(f"segmentations_test/{i}_{j}.png")