import os, sys
import torch
import pdb
path = os.getcwd()
os.chdir(path)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

def init_dataloader(args):
    batch_size = args.batch_size
    operator_workers = 1
    if args.l96:
        from dataloader.dataloader_l96 import TrainingData, TestingData
        train_dataset_operator = TrainingData(
            args.x_len,
            train_size=args.training_size,
            noisy_scale=args.noisy_scale,
            train_operator=True,
        )
    elif args.kse:
        from dataloader.dataloader_ks import KSTrainingData
        noisy_scale = getattr(args, 'noisy_scale', 0.0)
        batches_per_epoch = getattr(args, 'batches_per_epoch', 1)
        n_crops = args.batch_size * batches_per_epoch
        train_dataset_operator = KSTrainingData(
            crop_T=args.x_len,
            data_folder=args.ks_data_train,
            train_size=args.training_size,
            train_operator=True,
            validation=False,
            noisy_scale=noisy_scale,
            n_crops=n_crops,
        )
    elif args.l63:
        from dataloader.dataloader_l63 import L63TrainingData
        noisy_scale = getattr(args, 'noisy_scale', 0.0)
        batches_per_epoch = getattr(args, 'batches_per_epoch', 1)
        n_crops = args.batch_size * batches_per_epoch
        train_dataset_operator = L63TrainingData(
            crop_T=args.x_len,
            data_folder=args.l63_data_train,
            train_size=args.training_size,
            train_operator=True,
            validation=False,
            noisy_scale=noisy_scale,
            n_crops=n_crops,
        )
    else:
        raise ValueError('Must pass one of --l96, --kse, or --l63')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset_operator, shuffle=True)
        train_sampler_operator = torch.utils.data.distributed.DistributedSampler(train_dataset_operator, shuffle=True)
    else:
        train_sampler = None
        train_sampler_operator = None
    train_loader_operator = torch.utils.data.DataLoader(train_dataset_operator, batch_size=batch_size, \
                    shuffle=(train_sampler_operator is None), num_workers=operator_workers, pin_memory=True, \
                    sampler=train_sampler_operator, drop_last=True, persistent_workers=(operator_workers > 0))

    for i in range(min(10, len(train_dataset_operator))):
        train_dataset_operator[i]

    return train_dataset_operator, train_loader_operator, train_sampler, train_sampler_operator
