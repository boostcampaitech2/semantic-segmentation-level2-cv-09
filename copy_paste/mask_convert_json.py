import argparse
import glob
import tqdm
from src.create_annotations import *
import random
import torch
import argparse

'''
coco format json으로 변경하는 파일입니다. 

'''


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# Label ids of the dataset
category_ids = {
    #"Background":0,
    "General trash": 1,
    "Paper": 2,
    "Paper pack": 3,
    "Metal": 4,
    "Glass": 5,
    "Plastic": 6,
    "Styrofoam": 7,
    "Plastic bag": 8,
    "Battery": 9,
    "Clothing": 10
}

# Define which colors match which categories in the images
category_colors = {
    # "(0, 0, 0)": 0, # Background
    "(128, 0, 0)": 1, # General trash
    "(0, 128, 0)": 2, # Paper
    "(128, 128, 0)": 3, # Paper pack
    "(0, 0, 128)": 4, # Metal
    "(128, 0, 128)": 5, # Glass
    "(0, 128, 128)": 6, # Plastic
    "(128, 128, 128)": 7, # Styrofoam
    "(64, 0, 0)": 8, # Plastic bag
    "(192, 0, 0)": 9, # Battery
    "(64, 128, 0)": 10 # Clothing
}

# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = []

# Get "images" and "annotations" info 
def images_annotations_info(dataset_path, annotation_id, image_id, annotations, images):
    
    for keyword in ['batch_04','batch_05', 'batch_06']:
        for mask_image in tqdm.tqdm(glob.glob(os.path.join(dataset_path, 'SegmentationCopy', keyword, "*.png"))):
            # The mask image is *.png but the original image is *.jpg.
            # We make a reference to the original file in the COCO JSON file
            original_file_name = os.path.join(keyword, os.path.basename(mask_image).replace('.png', '.jpg'))

            # Open the image and (to be sure) we convert it to RGB
            mask_image_open = Image.open(mask_image).convert("RGB")
            w, h = mask_image_open.size
            
            # "images" info 
            image = create_image_annotation(original_file_name, w, h, image_id)
            images.append(image)

            sub_masks = create_sub_masks(mask_image_open, w, h)
            for color, sub_mask in sub_masks.items():
                category_id = category_colors.get(color)
                if category_id==None:
                    continue

                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    if polygons[i].type == 'MultiPolygon':
                        continue
                        
                    else:
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                   
                    
                    
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)
                    annotations.append(annotation)
                    annotation_id += 1
            image_id += 1
    return images, annotations, annotation_id

def get_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparmeter
    parser.add_argument('--main_json', type=str, default='train_all.json', help='set main json file to add with copy_paste json file (default: train_all.json)')
    parser.add_argument('--mode', type=str, default='add', help='Yon can choose add or self (default : add)')

    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    # fix seed
    seed_everything(42)
    
    # arguments
    args = get_args()

    dataset_path = "/opt/ml/segmentation/input/data/"  

    if args.mode == 'add':  # 기존 json 파일과 합칠 때
        anns_file_path = dataset_path + args.main_json

        # Read annotations
        with open(anns_file_path, 'r') as f:
            coco_format = json.loads(f.read())
        
        # This id will be automatically increased as we go
        annotations = coco_format["annotations"]
        images = coco_format['images']
        annotation_id = len(annotations)
        image_id = len(images)


    elif args.mode == 'self': # json 파일을 합치지 않을 때

        # Get the standard COCO JSON format
        coco_format = get_coco_json_format()

        # This id will be automatically increased as we go
        annotations = []
        images = []
        annotation_id = 0
        image_id = 0

        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(dataset_path, annotation_id, image_id, annotations, images)

    with open("/opt/ml/segmentation/input/data/copy_paste.json","w") as outfile:
        json.dump(coco_format, outfile)
    

