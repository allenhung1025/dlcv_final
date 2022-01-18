import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from test_dataset import ImageDataset

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import csv
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
from tqdm import tqdm
import pandas as pd
from utils.data_utils import get_val_loader



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./output/sample_run_checkpoint.bin", help="model path")
    parser.add_argument("--normalize_weight", action="store_true")

    opt = parser.parse_args()
    print(opt)
    data_root = '../food_data'
    #transforms
    #define the model and load the checkpoint
    config = CONFIGS['ViT-B_16']
    model = VisionTransformer(config, 448, zero_head=True, num_classes=1000)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model_path)['model'])

    if opt.normalize_weight:
        weight = model.part_head.weight.data
        denominator = torch.sqrt(torch.sum(weight ** 2, dim = 1))
        denominator.unsqueeze_(1)
        denominator = denominator.repeat(1, 768)
        new_weight = weight / denominator
        model.part_head.weight = nn.Parameter(new_weight)
    model.eval()

    print('model loaded')
    #define the dataset

    test_dataloaders = get_val_loader(data_root)
    print('total {} task to test'.format(len(test_dataloaders)))

    
    for test_loader in test_dataloaders[1:]:
        
        correct = 0.
        total   = 0.
        for batch in tqdm(test_loader):
            batch = (batch['img'].cuda(), batch['label'])
            x, y = batch

            output = model(x)
            pred = output.argmax(dim = 1).item()
            if pred == y:
                correct += 1
            total += 1
        
        print('acc:', correct / total)

            
