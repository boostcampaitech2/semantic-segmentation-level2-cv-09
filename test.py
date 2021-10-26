import argparse
import os
from tqdm import tqdm
import torch
import utils
import numpy as np
import pandas as pd

from importlib import import_module
import albumentations as A

def test(data_dir, model_dir, args):

    # -- settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = args.size
    resize_module = A.Compose([A.Resize(size, size)])
    
    # -- dataset
    test_dataset_module = getattr(import_module("dataset"), args.dataset) # default: BaseDataset
    test_dataset = test_dataset_module(
        data_dir = data_dir,
        ann_file = "test.json",
        train = False
    )

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation) # dafualt: BaseAugmentation
    transform = transform_module()
    test_dataset.set_transform(transform)

    # -- data loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size = args.batch_size,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    # -- model load
    model = torch.load(os.path.join(model_dir, args.model_name)).to(device)

    # -- inference

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (images, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 * 512)
            outs = model(torch.stack(images).to(device))
            oms = torch.argmax(outs, dim=1).detach().cpu().numpy()

            # resize (256 * 256)
            temp_mask = []
            for image, mask in zip(np.stack(images), oms):
                resized = resize_module(image=image, mask=mask)
                mask = resized['mask']
                temp_mask.append(mask)
            
            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])

    file_names = [y for x in file_name_list for y in x]


    # -- create submission 
    submission = pd.read_csv('./sample_submission.csv', index_col=None)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(args.model_dir, "best_model.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument()

    parser.add_argument('--model_dir', type=str, default='model/exp', help='model save dir (default: model/exp)')
    parser.add_argument('--model_name', type=str, default='best.pth', help='model name (default: best.pth)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset type (default: BaseDataset)')
    parser.add_argument('--augmentation', type=str, default='TestAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--num_workers', type=int, default=4, help='worker size for dataloader (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--size', type=int, default=256, help='test resolution size (defualt: 256)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/'))

    args = parser.parse_args()
    print(args)
    test(args.data_dir, args.model_dir, args)

    # trainer.model_test(args)
