from dataset import BaseDataset
import torch
import os
import cv2
import numpy as np
import pandas as pd
from shutil import copyfile
import scipy.ndimage

if __name__ == "__main__":
    csv_file = "../../mmsegmentation/work_dirs/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/output.csv"
    data_dir = "../../input/data"
    output_dir = "../../input/data/mmseg/train"
    target_image_dir = os.path.join(output_dir, "image")
    target_mask_dir = os.path.join(output_dir, "mask")

    submission = pd.read_csv(csv_file, index_col=None)

    image_ids = submission["image_id"].values
    masks = submission["PredictionString"].values

    idx = 4117
    for image_id, mask in zip(image_ids, masks):
    
        # image load
        image = cv2.imread(os.path.join(data_dir, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # mask load
        mask = list(map(int, mask.split()))
        mask = np.array(mask)
        mask = np.reshape(mask, (-1, 256))
        mask = scipy.ndimage.zoom(mask, 2, order=0) # resize 512, 512
        
        cv2.imwrite(os.path.join(target_mask_dir, "{0:05d}.png".format(idx)), mask)
        copyfile(os.path.join(data_dir, image_id), os.path.join(target_image_dir, "{0:05d}.jpg".format(idx)))
        
        idx += 1