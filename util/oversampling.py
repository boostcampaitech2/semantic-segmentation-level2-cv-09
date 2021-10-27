import argparse
import albumentations as A


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

def get_background():
    pass

def do_synthesis(background, patch):
    pass

def main(args):

    if args.merge_json:
        print("merge data with original json")
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--patch")
    parser.add_argument("--background")
    parser.add_argument("--num_output", type=int, default=500)
    parser.add_argument("--data_dir", type=str, default="opt/ml/segmentation/data/input")
    parser.add_argument("--json_path", type=str, default="train_all.json")
    parser.add_argument("--output_json", type=str, default="oversampled_train.json")
    parser.add_argument("--merge_json", default=True, action='store_true')
    args = parser.parse_args()

    main(args)

