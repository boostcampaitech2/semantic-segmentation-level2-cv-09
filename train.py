import argparse
import trainer
import os
import utils
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameter
    parser.add_argument('--model', type=str, default='FCN32s', help='architecture (default: FCN32s)')
    parser.add_argument('--backbone', type=str, default="resnet50", help="set backbone like encoder (default: resnet50)")
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss", help='loss function (default: CrossEntropyLoss)')
    parser.add_argument('--dataset', type=str, default='BaseDataset', help='dataset type (default: BaseDataset)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument('--val_json', type=str, default='val.json', help='default: val.json')
    parser.add_argument('--train_json', type=str, default='train.json', help='default: train.json')
    parser.add_argument('--train_augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--val_augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--num_workers', type=int, default=4, help='worker size for dataloader (default: 4)')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training (default: 8)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=1, help='validation term (default: 1)')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--save_every', default=True, action='store_true', help='save mode (default: True)')
    parser.add_argument('--val', default=True, action='store_true', help='using val set for validation (default: True)')
    parser.add_argument('--model_test', default=False, action='store_true', help='model test mode(True) (default: False)')
    parser.add_argument('--experimenter', default='kjy', help='your name initial (default: kjy)')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)
    if args.model_test:
        utils.model_test(args)
    else:
        trainer.train(args.data_dir, args.model_dir, args)

    # trainer.model_test(args)

