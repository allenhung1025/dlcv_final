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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../food_data/test", help="input directory")
    parser.add_argument("--sample_csv", type=str, default="../food_data/testcase/sample_submission_comm_track.csv", help="input directory")
    parser.add_argument("--output_csv", type=str, default="./output_common_track.csv", help="output csv")
    parser.add_argument("--model_path", type=str, default="./output/sample_run_checkpoint.bin", help="model path")
    parser.add_argument("--normalize_weight", action="store_true")

    opt = parser.parse_args()
    print(opt)

    #transforms
    test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                    transforms.CenterCrop((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    #define the model and load the checkpoint
    config = CONFIGS['ViT-B_16']
    model = VisionTransformer(config, 448, zero_head=True, num_classes=1000)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.model_path)['model'], strict=False)

    if opt.normalize_weight:
        weight = model.part_head.weight.data
        denominator = torch.sqrt(torch.sum(weight ** 2, dim = 1))
        denominator.unsqueeze_(1)
        denominator = denominator.repeat(1, 768)
        new_weight = weight / denominator
        model.part_head.weight = nn.Parameter(new_weight)
    model.eval()

    #define the dataset
    test_dataset = ImageDataset(opt.input_dir, opt.sample_csv, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)

    #pred labels
    pred_labels = []

    #output csv
    output_df = pd.DataFrame()

    for image_id, img in tqdm(test_dataloader):
        image_id = image_id[0]
        img = img.cuda()
        output = model(img)
        pred = output.argmax(dim = 1).item()
        pred_labels.append(pred)

    output_df['image_id'] = test_dataset.image_ids
    output_df['label'] = pred_labels
    output_df.to_csv(opt.output_csv, index=False)
    print('writing to file {}'.format(opt.output_csv))
