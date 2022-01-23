import logging
from PIL import Image
import os

import torch

from torchvision           import transforms
from torch.utils.data      import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from torchsampler          import ImbalancedDatasetSampler


from .dataset     import CUB, CarsDataset, NABirds, dogs, INat2017, food
from .autoaugment import AutoAugImageNetPolicy

logger = logging.getLogger(__name__)




def get_loader(args):

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    
    train_transform=[transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.RandomCrop((448, 448)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    test_transform=[transforms.Resize((600, 600), Image.BILINEAR),
                                transforms.CenterCrop((448, 448)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    trainset = food(args.data_root , '../food_data/label2name.txt', train_transform, mode="train", image_type="all") 
    testsets = {image_type: food(args.data_root, '../food_data/label2name.txt',  test_transform, mode='val'  , image_type=image_type) for image_type in ['r', 'c', 'f']}        


    if args.local_rank == 0:
        torch.distributed.barrier()
    
    train_sampler = ImbalancedDatasetSampler(trainset) if args.balanced else RandomSampler(trainset)
    train_sampler = train_sampler if args.local_rank == -1 else DistributedSamplerWrapper(train_sampler)
    train_loader  = DataLoader(trainset,
                               sampler=train_sampler,
                               batch_size=args.train_batch_size,
                               num_workers=4,
                               drop_last=True,
                               pin_memory=True)
    test_loaders = {}
    for key, testset in testsets.items():
        test_sampler  = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
        test_loader   = DataLoader(testset,
                                   sampler=test_sampler,
                                   batch_size=args.eval_batch_size,
                                   num_workers=4,
                                   pin_memory=True) if testset is not None else None
        if test_loader is not None:
            test_loaders[key] = test_loader

    return train_loader, test_loaders if len(test_loaders.keys()) != 0 else None
