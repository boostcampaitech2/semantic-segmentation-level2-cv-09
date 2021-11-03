python train.py \
--model UPlusPlus_Efficient_b5 \
--epochs 200 \
--loss FocalLoss \
--val_json kfold_0_val.json \
--train_json kfold_0_train.json \
--train_augmentation CustomTrainAugmentation \
--batch_size 5