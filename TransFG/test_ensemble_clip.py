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
import clip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../food_data/test", help="input directory")
    parser.add_argument("--sample_csv", type=str, default="../food_data/testcase/sample_submission_comm_track.csv", help="input directory")
    parser.add_argument("--label2name", type=str, default="../food_data/label2name_en.txt", help="input directory")
    parser.add_argument("--output_csv", type=str, default="./output_common_track.csv", help="output csv")
    parser.add_argument("--image_type", type=str, default="c", help="output csv")
    parser.add_argument("--model_path", type=str, default="./output/sample_run_checkpoint.bin", help="model path")
    parser.add_argument("--checkpoint_clip", type=str, default="../CLIP/checkpoint/model_10.pt", help="fine-tuned checkpoint")
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
    model = nn.DataParallel(model)

    if opt.normalize_weight:
        weight = model.part_head.weight.data
        denominator = torch.sqrt(torch.sum(weight ** 2, dim = 1))
        denominator.unsqueeze_(1)
        denominator = denominator.repeat(1, 768)
        new_weight = weight / denominator
        model.part_head.weight = nn.Parameter(new_weight)
    model.eval()

    #define clip model
    clip_model, preprocess = clip.load("ViT-B/32", jit=False) #Must set jit=False for training
    checkpoint = torch.load(opt.checkpoint_clip)
    clip_model.load_state_dict(checkpoint['model_state_dict'])
    clip_model = nn.DataParallel(clip_model)
    clip_model.cuda().eval()


    #retrieve text and label from label2name file
    label2name_df = pd.read_csv(opt.label2name)
    image_en_classes = label2name_df['3'].tolist()

    text_descriptions = [f"This is a photo of {label}, a type of food." for label in image_en_classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()

    #compute text features
    with torch.no_grad():
        text_features = clip_model.module.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    

    #define the dataset
    test_dataset_transfg = ImageDataset(opt.input_dir, opt.sample_csv, test_transform)
    test_dataset_clip = ImageDataset(opt.input_dir, opt.sample_csv, preprocess)
    test_dataloader_transfg = DataLoader(test_dataset_transfg, batch_size = 4)
    test_dataloader_clip = DataLoader(test_dataset_clip, batch_size = 4)

    #pred labels
    pred_labels = []

    #output csv
    output_df = pd.DataFrame()

    for  _, (data1, data2) in tqdm(enumerate(zip(test_dataloader_transfg, test_dataloader_clip)), total = len(test_dataloader_transfg)):
        #TransFG
        image_id = data1[0][0]
        img = data1[1].cuda()
        output = model(img)

        output = output.detach().cpu()
        output = output.softmax(dim = -1)


        #clip
        with torch.no_grad():
            img = data2[1].cuda()
            image_features = clip_model.module.encode_image(img).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        text_probs = text_probs.cpu()
        output = (output + text_probs) / 2

        pred = output.argmax(dim = 1).tolist()
        pred_labels.extend(pred)

    output_df['image_id'] = test_dataset_clip.image_ids
    output_df['label'] = pred_labels
    output_df.to_csv(opt.output_csv, index=False)
    print('writing to file {}'.format(opt.output_csv))
