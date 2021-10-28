"""
get semantic segmentation annotations from coco data set.
"""
from PIL import Image
import imgviz
import argparse
import os
import tqdm
from pycocotools.coco import COCO
import shutil
import numpy as np
import random
import torch

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
 

#  input_dir = ../input/data/ 
#  split = train_all
 
def main(args):
    annotation_file = os.path.join(args.input_dir, '{}.json'.format(args.split))
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass', 'batch_04'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass', 'batch_05'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass', 'batch_06'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages', 'batch_04'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages', 'batch_05'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages', 'batch_06'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            img_origin_path = os.path.join(args.input_dir, img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'].replace('batch_01_vt','batch_04').replace('batch_02_vt','batch_05').replace('batch_03','batch_06'))
            seg_output_path = os.path.join(args.input_dir, 'SegmentationClass', img['file_name'].replace('.jpg', '.png').replace('batch_01_vt','batch_04').replace('batch_02_vt','batch_05').replace('batch_03','batch_06'))
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)
 
 
def get_args():
    # fix seed
    seed_everything(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../../input/data/", type=str,
                        help="coco dataset directory")
    parser.add_argument("--split", default="train_all", type=str,
                        help="train_all or train or val")
    return parser.parse_args()
 
 
if __name__ == '__main__':
    args = get_args()
    main(args)