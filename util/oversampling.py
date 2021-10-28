import argparse
import json
import albumentations as A
import tqdm
import os
import numpy as np
import cv2
from pycocotools.coco import COCO

def apply_augmentation(image, mask):
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomCrop(384, 384, p=0.5),
        A.RandomBrightnessContrast(),
    ])
    transformed = transforms(image=image, mask=mask)

    return transformed["image"], transformed["mask"]

def get_patches():
    pass

def id_to_objnum(coco, ann):
    anns = coco.getAnnIds(imgIds=ann['id'])
    print(anns)
    return len(coco.getAnnIds(imgIds=ann['id']))
    
def get_background(target_categories, num_output, path):
    """
        1. json file load
        2. json file load using coco library
        3. finding an image have single object
    """
    json_file = None
    with open(path, 'r') as f:
        json_file = json.load(f)

    coco_from = COCO(path)
    images = json_file["images"]
    single_obj_images = []
    for ann in images:
        if id_to_objnum(coco_from, ann) == 1:
            single_obj_images.append(ann)
    
    target_single_obj_images = []
    # for image in single_obj_images:

    pass

def do_synthesis(background, patch):
    pass

def main(args):

    get_background(None, 0, os.path.join(args.data_dir, args.json_path))
    if args.merge_json:
        print("merge data with original json")
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--patch")
    parser.add_argument("--background")
    parser.add_argument("--num_output", type=int, default=500)
    parser.add_argument("--json_path", type=str, default="train_all.json")
    parser.add_argument("--output_json", type=str, default="oversampled_train.json")
    parser.add_argument("--merge_json", default=True, action='store_true')

    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/'))
    args = parser.parse_args()

    main(args)

