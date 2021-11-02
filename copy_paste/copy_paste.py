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
import albumentations as A


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


def transform(mask, img):

    tfms_to_small = A.Compose([
        A.Resize(256, 256),
        A.PadIfNeeded(512, 512, border_mode=0),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512)
    ])

    tfms_to_big = A.Compose([
        A.CropNonEmptyMaskIfExists(256, 256, ignore_values=[0]),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.Resize(512, 512)
    ])

    tfms = A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512)
    ])

    area = np.sum(mask[mask!=0] != 0)
    if  area > 20000:
        transformed = tfms_to_small(image=img, mask=mask)
        mask = transformed['mask']
        img = transformed['image']
    elif area < 5000:
        transformed = tfms_to_big(image=img, mask=mask)
        mask = transformed['mask']
        img = transformed['image']  
    else:
        transformed = tfms(image=img, mask=mask)
        mask = transformed['mask']
        img = transformed['image']

    return mask, img

def get_image_ann_id(coco:COCO, image:dict):
    ann_ids = coco.getAnnIds(imgIds=image['id'])
    return ann_ids, len(ann_ids)


def get_target_image(target_categories:list, remove_target_categories:list, path:str, coco_from:COCO):
    """
        1. json file load
        2. json file load using coco library
        3. finding an image that has a single object that you're aiming for
    """
    target = []
    for t_cat in target_categories:
        target.append(category[t_cat])
    remove_target = []
    for r_cat in remove_target_categories:
        remove_target.append(category[r_cat])

    print("target class:", target)
    print("remove target class:", remove_target)

    category_count = []

    json_file = None
    with open(path, 'r') as f:
        json_file = json.load(f)

    images = json_file["images"]

    single_obj_images = []
    for image in images:
        ids, length = get_image_ann_id(coco_from, image)
        anns_info = coco_from.loadAnns(ids)
        ann_category = set([ann['category_id'] for ann in anns_info]) # target annotation이 있고, remove_target_categories가 없으면 이미지 추가
        
        if len(ann_category & set(target)) != 0 and len(ann_category & set(remove_target)) == 0:
            single_obj_images.append([image['file_name'], ids])
            category_count.extend(list(ann_category))

    for key, value in category.items():
        print(f'{key} : {category_count.count(value)}')

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

    mask_src, img_src = transform(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)
    

    # LSJ， Large_Scale_Jittering
    # if args.lsj:
    #     mask_src, img_src = Large_Scale_Jittering(mask_src, img_src, args.lsj_min, args.lsj_max)
    #     mask_main, img_main = Large_Scale_Jittering(mask_main, img_main, args.lsj_min, args.lsj_max)
    # else:
    #     # rescale mask_src/img_src to less than mask_main/img_main's size
    #     h, w, _ = img_main.shape
    #     mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return mask, img


def extract_patch(mask, coco:COCO, ids):
    '''
    패치들만 추출하는 함수입니다.
    '''

    anns_info = coco.loadAnns(ids)
    anns = [ann for ann in anns_info if ann['category_id'] in args.patch] 
    for i in range(len(anns)):
        mask[coco.annToMask(anns[i]) == 0] = 0 # patch에 없는 것들은 삭제
    mask = mask.astype(np.uint8)

    return mask

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
    target_path = get_target_image(args.patch, args.remove_patch, os.path.join(args.output_dir, args.json_path), coco)
    image_count = 0
    images_path = []

    for target, ids in target_path:
        images_path.append([os.path.join(JPEGs, target), ids])# target: batch_04/0001.jpg
    images_path = random.choices(images_path, k=args.aug_num)
    tbar = tqdm.tqdm(images_path, ncols=100)
    for image_path in tbar:
        # get source mask and img
 
        mask_src = np.asarray(Image.open(image_path[0].replace('.jpg', '.png').replace('JPEGImages', 'SegmentationClass')), dtype=np.uint8)
        if args.extract_patch:
            mask_src = extract_patch(mask_src, coco, image_path[1])
        img_src = cv2.imread(image_path[0])

        # random choice main mask/img
        mask_main_path = random.choices(images_path, k=1)[0][0]
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
    parser.add_argument("--lsj", default=False, type=bool, help="if use Large Scale Jittering")
    parser.add_argument("--lsj_min", default=0.2, type=float, help='recommend 0.2 ~ 0.4')
    parser.add_argument("--lsj_max", default=2, type=float, help='recommend 1.2 ~ 2')
    parser.add_argument("--json_path", default='train.json', type=str, help='recommend train.json')
    parser.add_argument("--patch", nargs="+", type=list, default=["Paper pack", "Battery", 'Clothing',"Glass" ])
    parser.add_argument("--remove_patch", nargs="+", type=list, default=["Paper", "Plastic bag"])
    parser.add_argument("--aug_num", nargs="+", type=int, default=1500)
    parser.add_argument("--extract_patch", type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
