"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import imgviz
import cv2
import argparse
import os
import numpy as np
import tqdm
import random
import torch
from pycocotools.coco import COCO
import json

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

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img


def get_image_ann_id(coco:COCO, image:dict):
    ann_ids = coco.getAnnIds(imgIds=image['id'])
    return ann_ids, len(ann_ids)


def get_single_ann_image(target_categories:list, path:str, coco_from:COCO):
    """
        1. json file load
        2. json file load using coco library
        3. finding an image that has a single object that you're aiming for
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
        ids, length = get_image_ann_id(coco_from, image)
        if True : # length == 1:
            anns_info = coco_from.loadAnns(ids)
            for ann in anns_info: # 한개라도 target annotation이 있으면 이미지 추가
                if ann["category_id"] in target:
                    single_obj_images.append(image['file_name'])
                    break

    return single_obj_images


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    # img_source에서 mask 값이 0이 아닌 픽셀만 연산 -> 마스크만 추출
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
    # 마스크를 img_main 사이즈에 맞춰 resize
    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    # img_main에 마스크를 집어넣을 부분만 add 연산을 통해 추출
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    # img_main = img_main - 마스크에 해당하는 원래 있던 부분 + 새로운 마스크 부분
    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_main


def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


def Large_Scale_Jittering(mask, img, min_scale=0.2, max_scale=2):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop


def copy_paste(mask_src, img_src, mask_main, img_main):
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ， Large_Scale_Jittering
    if args.lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src, args.lsj_min, args.lsj_max)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main, args.lsj_min, args.lsj_max)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img


def main(args):
    # fix seed
    seed_everything(42)

    # input path
    segclass = os.path.join(args.input_dir, 'SegmentationClass')
    JPEGs = os.path.join(args.input_dir, 'JPEGImages')

    # output_dir = ../input/data/
    # input_dir = ../input/data/

    # create output path
    os.makedirs(os.path.join(args.output_dir, 'SegmentationCopy', 'batch_04'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'batch_04'), exist_ok=True)
 
    
    # -- get target image path
    coco= COCO(os.path.join(args.input_dir, args.json_path))
    target_path = get_single_ann_image(args.patch, os.path.join(args.output_dir, args.json_path), coco)
    image_count = 0

  
    images_path = list(os.path.join(JPEGs, target) for target in target_path) # target: batch_04/0001.jpg
    tbar = tqdm.tqdm(images_path, ncols=100)
    for image_path in tbar:
        # get source mask and img
        mask_src = np.asarray(Image.open(image_path.replace('.jpg', '.png').replace('JPEGImages', 'SegmentationClass')), dtype=np.uint8)
        img_src = cv2.imread(image_path)

        # random choice main mask/img
        mask_main_path = np.random.choice(images_path)
        mask_main = np.asarray(Image.open(mask_main_path.replace('.jpg', '.png').replace('JPEGImages', 'SegmentationClass')), dtype=np.uint8)
        img_main = cv2.imread(os.path.join(mask_main_path))

        # Copy-Paste data augmentation
        mask, img = copy_paste(mask_src, img_src, mask_main, img_main)

        mask_filename = "copy_paste_" + f'{image_count:0>4}.png'
        img_filename = mask_filename.replace('.png', '.jpg')
        save_colored_mask(mask, os.path.join(args.output_dir, 'SegmentationCopy', 'batch_04', mask_filename))
        cv2.imwrite(os.path.join(args.output_dir, 'batch_04', img_filename), img)
        image_count+=1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../../input/data/", type=str,
                        help="input annotated directory")
    parser.add_argument("--output_dir", default="../../input/data/", type=str,
                        help="output dataset directory")
    parser.add_argument("--lsj", default=True, type=bool, help="if use Large Scale Jittering")
    parser.add_argument("--lsj_min", default=0.2, type=float, help='recommend 0.2 ~ 0.4')
    parser.add_argument("--lsj_max", default=2, type=float, help='recommend 1.2 ~ 2')
    parser.add_argument("--json_path", default='train.json', type=str, help='recommend train.json')
    parser.add_argument("--patch", nargs="+", type=list, default=["Paper pack", "Battery", "Plastic", 'Clothing',"Glass" ])

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
