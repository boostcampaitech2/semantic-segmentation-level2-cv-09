from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import dataset
import utils
import torch
import torch.nn as nn
import numpy as np
import os
import json
import wandb

import segmentation_models_pytorch as smp
from importlib import import_module

class_labels = {
    0: "Background",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing"
}

# util function for generating interactive image mask from components
def wb_mask(bg_img, pred_mask, true_mask):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : class_labels},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : class_labels}})

def train(data_dir, model_dir, args): # data_dir, model_dir, args

    # -- settings
    utils.set_seed(42)
    save_dir = utils.increment_path(os.path.join(model_dir, args.name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(save_dir)
    
    # -- dataset
    train_dataset_module = getattr(import_module("dataset"), args.dataset) # default: BaseDataset
    train_dataset = train_dataset_module(
        data_dir = data_dir,
        ann_file = "train.json",
        train = True
    )
    if args.val:
        val_dataset_module = getattr(import_module("dataset"), args.dataset) # default: BaseDataset
        val_dataset = val_dataset_module(
            data_dir = data_dir,
            ann_file = "val.json",
            train = True
        )
    num_classes = train_dataset.num_classes

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.train_augmentation) # dafualt: BaseAugmentation
    transform = transform_module()
    train_dataset.set_transform(transform)
    if args.val:
        val_transform_module = getattr(import_module("dataset"), args.val_augmentation)
        val_transform = val_transform_module()
        val_dataset.set_transform(val_transform)
    
    # -- data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        drop_last = True
    )
    if args.val:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
            drop_last = True
        )
    # -- model
    model_module = getattr(import_module("models"), args.model)
    model = model_module(
        backbone=args.backbone,
        num_classes=11
    ).to(device)

    # -- loss & metric
    # criterion = nn.CrossEntropyLoss()
    # criterion = smp.losses.FocalLoss(mode="multiclass")

    criterion_module = getattr(import_module("loss"), args.loss)
    criterion = criterion_module()

    optimizer_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = optimizer_module(model.parameters(), lr = args.lr)

    # scheduler_moudler = getattr(import_module("scheduler"), args.scheduler)
    # scheduler = scheduler_moudler()
    # scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-3)

    # -- logging
    wandb.init(project='trash_segmentation', entity='cv-09-segmentation', name = "_".join([args.experimenter, args.model, args.backbone, args.optimizer, str(args.epochs)]))
    wandb.config = args
    wandb.watch(model)

    utils.write_json(save_dir, "config.json", vars(args))

    # -- loop
    best_val_mIoU = 0

    for epoch in range(args.epochs):
        # train loop
        model.train()

        hist = np.zeros((num_classes, num_classes))
        for step, (images, masks, _) in enumerate(train_loader):
            images = torch.stack(images)
            masks = torch.stack(masks).long()

            # to device
            images, masks = images.to(device), masks.to(device)

            # inference
            outputs = model(images)

            # loss
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # detach
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = utils.add_hist(hist, masks, outputs, num_classes)
            acc, acc_cls, mIoU, fwavacc, IoU = utils.label_accuracy_score(hist)
            current_lr = utils.get_lr(optimizer)

            if (step + 1) % args.log_interval == 0:
                msg = f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)} lr {current_lr}'
                print(msg)
                utils.logging(save_dir, "log.txt", msg)
                wandb.log({"epoch":epoch, "loss":round(loss.item(), 4), "mIoU":round(mIoU, 4), "lr":current_lr})

        # scheduler.step()

        if args.val == False:
            utils.save_model(model, save_dir, f"epoch{epoch}.pth")

        # val loop
        if (epoch + 1) % args.val_every == 0 and args.val:
            model.eval()

            with torch.no_grad():
                total_loss = 0
                cnt = 0
                
                hist = np.zeros((num_classes, num_classes))
                for step, (images, masks, _) in enumerate(val_loader):
                    
                    images = torch.stack(images)       
                    masks = torch.stack(masks).long()  

                    images, masks = images.to(device), masks.to(device)            
                    
                    # device 할당
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    total_loss += loss
                    cnt += 1
                    
                    outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()
                    
                    hist = utils.add_hist(hist, masks, outputs, num_classes)
                
                # val set mIoU 계산
                acc, acc_cls, mIoU, fwavacc, IoU = utils.label_accuracy_score(hist)
                IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , val_dataset.category_names)]
                
                avrg_loss = total_loss / cnt
                msg = f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                        mIoU: {round(mIoU, 4)}'
                print(msg)
                utils.logging(save_dir, "log.txt", msg)

                msg = f'IoU by class : {IoU_by_class}'
                print(msg)
                utils.logging(save_dir, "log.txt", msg)

                for iou in IoU_by_class:
                    wandb.log(iou)

                wandb.log({"epoch":epoch, "avg loss":round(avrg_loss.item(), 4), "acc":round(acc, 4), "mIoU":round(mIoU, 4)})

                # wandb에 이미지를 보내는 기능입니다.
                mask_list=[] 
                for image, pred, mask in zip(images, outputs, masks):
                    mask_list.append(wb_mask(image, pred, mask))
                wandb.log({"predictions" : mask_list})

            # 이전 epoch 보다 mIoU 증가한 경우 모델 저장
            if mIoU < best_val_mIoU:
                print(f"Best performance at epoch: {epoch + 1}")
                utils.logging(save_dir, "log.txt", f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {save_dir}")
                utils.logging(save_dir, "log.txt", f"Save model in {save_dir}")
                best_val_mIoU = mIoU
                utils.save_model(model, save_dir, "best.pth")

            # epoch마다 모델 저장
            if args.save_every:
                utils.save_model(model, save_dir, f"epoch{epoch + 1}.pth")
            else:
                utils.save_model(model, save_dir, f"last.pth")

