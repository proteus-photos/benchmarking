from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from ultralytics import YOLO
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from utils import clip_to_image
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

X = 0
Y = 1
W = 2
H = 3

def combine(mask1, mask2):
    x = min(mask1["bbox"][X], mask2["bbox"][X])
    y = min(mask1["bbox"][Y], mask2["bbox"][Y])
    w = max(mask1["bbox"][X] + mask1["bbox"][W], mask2["bbox"][X] + mask2["bbox"][W]) - x
    h = max(mask1["bbox"][Y] + mask1["bbox"][H], mask2["bbox"][Y] + mask2["bbox"][H]) - y

    crop_x = min(mask1["crop_box"][X], mask2["crop_box"][X])
    crop_y = min(mask1["crop_box"][Y], mask2["crop_box"][Y])
    crop_w = max(mask1["crop_box"][X] + mask1["crop_box"][W], mask2["crop_box"][X] + mask2["crop_box"][W]) - crop_x
    crop_h = max(mask1["crop_box"][Y] + mask1["crop_box"][H], mask2["crop_box"][Y] + mask2["crop_box"][H]) - crop_y
    
    return {
            "segmentation": mask1["segmentation"] | mask2["segmentation"],
            "area": mask1["area"] + mask2["area"] - (mask1["segmentation"] & mask2["segmentation"]).sum(),
            "bbox": [x, y, w, h],
            "predicted_iou": (mask1["predicted_iou"] * mask1["area"] + mask2["predicted_iou"] * mask2["area"]) / (mask1["area"] + mask2["area"]),
            "point_coords": [[
                (mask1["point_coords"][0][X] * mask1["area"] + mask2["point_coords"][0][X] * mask2["area"]) / (mask1["area"] + mask2["area"]),
                (mask1["point_coords"][0][Y] * mask1["area"] + mask2["point_coords"][0][Y] * mask2["area"]) / (mask1["area"] + mask2["area"])
            ]],
            "stability_score": (mask1["stability_score"] * mask1["area"] + mask2["stability_score"] * mask2["area"]) / (mask1["area"] + mask2["area"]),
            "crop_box": [crop_x, crop_y, crop_w, crop_h]
        }

def box_mask_intersection(maskA, maskB, threshold=0.05):
    mask = maskA["segmentation"]
    box = maskB["bbox"]
    x, y, w, h = clip_to_image(box, mask.shape[1], mask.shape[0])

    # box = [x, y, w, h]
    # print(box, mask.shape)

    # # check borders for segmentation intersection
    # # NOTE: sometimes predicted bbox extends beyond image, so preprocess and cut

    # return any((np.any(mask[box[Y]:box[Y]+box[H], box[X]]),
    #             np.any(mask[box[Y], box[X]:box[X]+box[W]]),
    #             np.any(mask[box[Y]:box[Y]+box[H], box[X]+box[W]]),
    #             np.any(mask[box[Y]+box[H], box[X]:box[X]+box[W]])))

    return maskA["segmentation"][y:y+h, x:x+w].sum() > maskA["area"] * threshold

def intersection_over_smaller(boxA, boxB):
    # xA = max(boxA["bbox"][X], boxB["bbox"][X])
    # yA = max(boxA["bbox"][Y], boxB["bbox"][Y])
    # xB = min(boxA["bbox"][X] + boxA["bbox"][W], boxB["bbox"][X] + boxB["bbox"][W])
    # yB = min(boxA["bbox"][Y] + boxA["bbox"][H], boxB["bbox"][Y] + boxB["bbox"][H])

    # interArea = max(0, xB - xA) * max(0, yB - yA)
    # boxAArea = boxA["bbox"][W] * boxA["bbox"][H]
    # boxBArea = boxB["bbox"][W] * boxB["bbox"][H]

    interArea = (boxA["segmentation"] & boxB["segmentation"]).sum()
    boxAArea = boxA["segmentation"].sum()
    boxBArea = boxB["segmentation"].sum()

    return interArea / min(boxAArea, boxBArea)

def supress_subsets(masks, n_final_masks=4):
    # combines masks based on high ios scores until either there are n_final_masks left or all masks are non-intersecting
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

        # if there is only slight overlap, break
        # REMEMBER IF YOU REMOVE THIS YOU STILL NEED TO KEEP == 0 CONDITION else it will try to combine first mask with itself
        if ios_matrix[mask1_index, mask2_index] < 0.1:
            break


        # combine the two dictionaries based on weighted average of area store in mask1_index, delete mask2_index

        masks[mask1_index] = combine(masks[mask1_index], masks[mask2_index])

        ios_matrix[mask1_index] = [intersection_over_smaller(masks[mask1_index], masks[i]) if mask1_index!=i else 0 for i in range(len(masks))]
        ios_matrix[:, mask1_index] = ios_matrix[mask1_index]

        masks.pop(mask2_index)
        ios_matrix = np.delete(np.delete(ios_matrix, mask2_index, axis=0), mask2_index, axis=1)

def suppress_small_masks(masks, area=600):
    # if takes all chhota masks, sees its border's interesection with bigger masks and combines if match
    masks.sort(key=lambda x: x["area"])
    i = 0
    while i < len(masks):
        mask = masks[i]
        if mask["area"] > area:
            i+=1
            continue
        j = i+1

        while j < len(masks):
            # check intersection of mask with all bigger masks
            if box_mask_intersection(mask, masks[j]):
                masks[j] = combine(mask, masks[j])
                masks.pop(i)
                break
            j+=1
        if j == len(masks):
            masks.pop(i)  # if no match found udaa do bc
        

class SAMSegmenter:
    def __init__(self):
        model_type = "vit_b"
        sam_checkpoint = "./weights/sam_vit_b.pth"

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            mobile_sam,
            points_per_side=16,
            min_mask_region_area=512
        )


    def segment(self, image):

        masks = self.mask_generator.generate(np.array(image))

        if len(masks) == 0:
            return [{
                'segmentation': np.ones((image.size[1], image.size[0]), dtype=bool),
                'area': image.size[0] * image.size[1],
                'bbox': [0, 0, image.size[1], image.size[0]],
                'predicted_iou': -1,
                'point_coords': [image.size[1]//2, image.size[0]//2],
                'stability_score': -1,
                'crop_box': [0, 0, image.size[1], image.size[0]]
            }]
        supress_subsets(masks, n_final_masks=5)

        # try and combine all masks smaller than 2% of the image area with bigger masks
        min_area = (image.size[1]*image.size[0]) * 0.02
        suppress_small_masks(masks, min_area)

        masks.append({
            "segmentation": np.ones((image.size[1], image.size[0]), dtype=bool),
            "area": 1,
            "bbox": [0, 0, image.size[0], image.size[1]]
        })

        return masks

class MaskRCNNSegmenter:
    def __init__(self):
        self.mask_generator = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.mask_generator.to(device=device)
        self.mask_generator.eval()

    def segment(self, image):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        print(image.max(), image.min())
        masks = self.mask_generator(image)[0]

        masks["bbox"] = masks.pop("boxes").detach().cpu().numpy()
        masks["segmentation"]  = masks.pop("masks").detach().cpu().squeeze(1).numpy() > 0.5
        masks["area"] = masks["segmentation"].sum(axis=(1, 2))
        masks["label"] = masks.pop("labels").detach().cpu().numpy()
        masks["score"] = masks.pop("scores").detach().cpu().numpy()

        masks = [dict(zip(masks.keys(), t)) for t in zip(*masks.values())]
        return masks
    
class YOLOSegmenter:
    def __init__(self):
        self.mask_generator = YOLO("./weights/yolov8x.pt")
        self.mask_generator.to(device=device)

    def segment(self, image):
        # input_image = torch.from_numpy(np.array(image.resize((320, 320)))).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        results = self.mask_generator(image, verbose=False)[0].cpu().numpy()
        if len(results) == 0:
            return [{
                "segmentation": np.ones((image.size[1], image.size[0]), dtype=bool),
                "area": 1,
                "bbox": [0, 0, image.size[0], image.size[1]]
            }]
        masks = {}
        masks["bbox"] = results.boxes.xywh
        masks["bbox"][:, :2] -= masks["bbox"][:, 2:] / 2

        masks["segmentation"]  = [np.zeros(image.size, dtype=bool).T for _ in range(len(results))]

        results.save(filename=f"test.jpg")

        for i, bbox in enumerate(masks["bbox"]):
            bbox = [int(x) for x in bbox]
            masks["segmentation"][i][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = True

        # masks["segmentation"]  = [np.array(Image.fromarray(mask.repeat(3, axis=-1)).resize(image.size))[..., 0] > 0.5
        #                           for mask in results.masks.data[..., None].astype(np.uint8)*255]

        masks["area"] = [mask.sum()/mask.size for mask in masks["segmentation"]]

        masks = [dict(zip(masks.keys(), t)) for t in zip(*masks.values())]
        masks.append({
            "segmentation": np.ones((image.size[1], image.size[0]), dtype=bool),
            "area": 1,
            "bbox": [0, 0, image.size[0], image.size[1]]
        })
        return masks
     
if __name__ == "__main__":
    s = SAMSegmenter()
    images = []
    for path in os.listdir('images_test'):
        image = Image.open(f'images_test/{path}').convert("RGB")
        images.append(image)

    shutil.rmtree('./segmentations_test', ignore_errors=True)
    os.mkdir('./segmentations_test')

    for i, image in tqdm(enumerate(images)):
        masks = s.segment(image)
 
        if len(masks) == 0:
            continue
        
        for j, ann in enumerate(masks):
            segmented_img = np.array(image.copy())
            m = ann['segmentation']
            segmented_img[m] = np.array([0, 0, 255])
            save_img = Image.fromarray(segmented_img)
            save_img.save(f"segmentations_test/{i}_{j}.png")