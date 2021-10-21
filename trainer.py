import utils
import torch
import torch.nn as nn
import numpy as np
import os
import segmentation_models_pytorch as smp
from importlib import import_module

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
    )
    if args.val:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=utils.collate_fn,
        )
    # -- model
    model_module = getattr(import_module("models"), args.model)
    model = model_module(num_classes=11).to(device)

    # -- loss & metric
    # criterion = nn.CrossEntropyLoss()
    criterion = smp.losses.FocalLoss(mode="multiclass")
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    
    # -- logging

    # -- loop
    best_val_loss = np.inf

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

            if (step + 1) % args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{len(train_loader)}], \
                        Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')

        if args.val == False:
            utils.save_model(model, save_dir, f"epoch{step}.pth")

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
                
                acc, acc_cls, mIoU, fwavacc, IoU = utils.label_accuracy_score(hist)
                IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , val_dataset.category_names)]
                
                avrg_loss = total_loss / cnt
                print(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, \
                        mIoU: {round(mIoU, 4)}')
                print(f'IoU by class : {IoU_by_class}')

            if avrg_loss < best_val_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {save_dir}")
                best_val_loss = avrg_loss
                utils.save_model(model, save_dir, "best.pth")

            utils.save_model(model, save_dir, f"epoch{step}.pth")

def model_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_module = getattr(import_module("models"), args.model)
    model = model_module(num_classes=11)

    input = torch.randn([8, 3, 512, 512])
    print("input shape:", input.shape)
    output = model(input).to(device)
    print("output shape: ", output.shape)