import argparse
import json
import albumentations as A
from tqdm import tqdm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import copy
import torch
import random

category = {
    "Background": 0,
    "General trash":1,
    "Paper":2,
    "Paper pack":3,
    "Metal":4,
    "Glass":5,
    "Plastic":6,
    "Styrofoam":7,
    "Plastic bag":8,
    "Battery":9,
    "Clothing":10,
}

def apply_augmentation(image, mask):
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.5),
        A.Resize(512, 512)
    ])
    transformed = transforms(image=image, mask=mask)

    return transformed

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def id_to_objnum(coco, image):
    ann_ids = coco.getAnnIds(imgIds=image['id'])
    return ann_ids, len(ann_ids)

def get_source(target_categories, path, coco_from, num_output=None):
    """
        1. json file load
        2. json file load using coco library
        3. finding an image have single object
    """
    target = []
    for t_cat in target_categories:
        target.append(category[t_cat])
    print("target class:", target)

    json_file = None
    with open(path, 'r') as f:
        json_file = json.load(f)

    images = json_file["images"]

    single_obj_images = []
    for image in images:
        ids, length = id_to_objnum(coco_from, image)
        if length == 1:
            ann_info = coco_from.loadAnns(ids)
            if ann_info[0]["category_id"] in target:
                single_obj_images.append(ids[0])
    
    masks = coco_from.loadAnns(single_obj_images)
    image_ids = [info["image_id"] for info in masks]
    target_single_obj_images = coco_from.loadImgs(image_ids)

    return target_single_obj_images, masks

def do_synthesis(patch, background):
    patch_image = patch[0]["image"]
    patch_mask = patch[0]["mask"]

    background_image = background[0]["image"]
    background_mask = background[0]["mask"]

    # cv2.imwrite('patch.jpg', patch_image)
    # cv2.imwrite('patch_mask.jpg', patch_mask)
    # cv2.imwrite('background.jpg', background_image)
    # cv2.imwrite('background_mask.jpg', background_mask)

    # image 합성
    target_mask = np.asarray(copy.deepcopy(patch_mask), dtype=np.uint8)
    target_mask[target_mask == 0] = 255
    target_mask[target_mask < 255] = 0
    # cv2.imwrite("target_mask.jpg", target_mask)

    masked_bg = cv2.bitwise_and(background_image, background_image, mask=target_mask)
    masked_p = cv2.bitwise_and(patch_image, patch_image, mask=patch_mask)
    # cv2.imwrite("masked_bg.jpg", masked_bg)
    # cv2.imwrite("masked_p.jpg", masked_p)

    new_image = cv2.add(masked_bg, masked_p)
    # cv2.imwrite("new.jpg", new_image)

    # mask 합성
    masked_bg = cv2.bitwise_and(background_mask, background_mask, mask=target_mask)
    new_mask = cv2.add(masked_bg, patch_mask)

    return new_image, new_mask
    # cv2.imwrite("new_mask.jpg", new_mask)

def main(args):
    # -- setting
    coco_from = COCO(os.path.join(args.data_dir, args.json_path))
    set_seed(42)
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)

    patches_source, patches_masks = get_source(args.patch, os.path.join(args.data_dir, args.json_path), coco_from)
    background_source, background_masks = get_source(args.background, os.path.join(args.data_dir, args.json_path), coco_from)

    print("Extract patch images...")
    patches = []
    for image_info, mask_info in tqdm(zip(patches_source, patches_masks), total=(len(patches_source))):
        image = cv2.imread(os.path.join(args.data_dir, image_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        mask = np.zeros((image_info["height"], image_info["width"]))
        pixel_value = mask_info["category_id"]
        mask[coco_from.annToMask(mask_info) == 1] = pixel_value
        mask = mask.astype(np.uint8)

        transformed = apply_augmentation(image, mask)
        patches.append(transformed)

    print("Extract background images...")
    background = []
    for image_info, mask_info in tqdm(zip(background_source, background_masks), total=(len(background_source))):
        image = cv2.imread(os.path.join(args.data_dir, image_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        mask = np.zeros((image_info["height"], image_info["width"]))
        pixel_value = mask_info["category_id"]
        mask[coco_from.annToMask(mask_info) == 1] = pixel_value
        mask = mask.astype(np.uint8)

        transformed = apply_augmentation(image, mask)
        background.append(transformed)

    print("Synthesis background and patch...")
    print("patches number:", len(patches))
    print("background number:", len(background))
    image_id = 60000
    for i in tqdm(range(args.num_output)):

        random_patch = np.random.choice(patches, 1)
        random_background = np.random.choice(background, 1)

        new_image, new_mask = do_synthesis(random_patch, random_background)
        cv2.imwrite(os.path.join(args.image_dir, "{0:05d}.jpg".format(image_id)), cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(args.mask_dir, "{0:05d}.png".format(image_id)), new_mask)
        image_id += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--patch", nargs="+", type=list, default=["Paper pack", "Battery", "Plastic"])
    parser.add_argument("--background", nargs="+", type=list, default=["Battery", "Clothing", "Metal", "General trash"])
    parser.add_argument("--num_output", type=int, default=500)
    parser.add_argument("--json_path", type=str, default="train_all.json")
    parser.add_argument("--output_json", type=str, default="oversampled_train.json")
    parser.add_argument("--merge_json", default=True, action='store_true')

    parser.add_argument("--image_dir", type=str, default="output/image")
    parser.add_argument("--mask_dir", type=str, default="output/mask")
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/'))
    args = parser.parse_args()

    main(args)

