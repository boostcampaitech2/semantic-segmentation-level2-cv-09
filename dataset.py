from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

data_dir = '../input/data'
ann_file_path = os.path.join(data_dir, 'train_all.json')

class BaseDataset(Dataset):
    def __init__(self, data_dir, ann_file, train=True, transform=None):
        super().__init__()
        self.train = train
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.transform = transform
        self.coco = COCO(os.path.join(data_dir, ann_file)) # pycocotools로 json 파일 load

        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)
        self.category_names = ["Background"] + [cat['name'] for cat in self.cats]
        self.num_classes = len(self.category_names)
        # print(self.cat_ids, self.num_classes)
        # print(self.cats)
        # print(self.category_names)

    def __getitem__(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        images = cv2.imread(os.path.join(self.data_dir, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.train == True:
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            masks = np.zeros((image_infos["height"], image_infos["width"])) # 비어있는 mask 생성
            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True) # 해당 이미지의 모든 annotation load
            for i in range(len(anns)):
                class_name = self.get_class_name(anns[i]['category_id'], self.cats)
                pixel_value = self.category_names.index(class_name)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.uint8)

            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]

            return images, masks, image_infos

        if self.train == False:
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, image_infos

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

    def get_class_name(self, class_id, cats):
        for i in range(len(cats)):
            if cats[i]['id']==class_id:
                return cats[i]['name']
        return "None"

    def set_transform(self, transform):
        self.transform = transform

class TestAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2(),
        ])
    def __call__(self, image):
        return self.transform(image=image)

class BaseAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            ToTensorV2(),
        ])
    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)

class CustomTrainAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.Perspective(p=0.5),
            A.ToGray(always_apply=False, p=0.2),
            A.OneOf(
                [
                    A.RandomBrightness(p=1.0),
                    A.RandomGamma(p=1.0),
                ],
                p=0.5,
            ),
            ToTensorV2(),
        ])
    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)