import glob
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, label2name, transforms_=None, mode="train", image_type="c"):

        """
        root: data directory with 1000 subdirectories
        transform_: transformation.
        label2name: label2name.
        mode: train, val.
        image_type: all, c, f, r.
        """

        self.transform = transforms.Compose(transforms_)
        self.image_type = image_type
        self.image_files = [] 

        # retrive the image labels from freq, common or rare.
        label2name_df = pd.read_csv(label2name, sep=" ", header=None)
        if image_type != "all":
            index = (label2name_df[1] == image_type)
            retrieved_label2name_df = label2name_df[index]
            image_classes= retrieved_label2name_df[0].tolist()
            image_classes_chinese = retrieved_label2name_df[2].tolist() 
        else:
            image_classes = [i for i in range(1000)]
            image_classes_chinese = label2name_df[2].tolist() 

        #self.image_files [[image_path, label, chinese_label], ...]
        directory = os.path.join(root, mode)
        for image_cls, image_cls_chinese in zip(image_classes, image_classes_chinese):
            image_cls_dir = os.path.join(directory, str(image_cls))
            for image in os.listdir(image_cls_dir):
                image_file = os.path.join(image_cls_dir, image)
                self.image_files.append([image_file, image_cls, image_cls_chinese])


    def __getitem__(self, idx):

        img = Image.open(self.image_files[idx][0])
        img = self.transform(img)
        label = self.image_files[idx][1]
        chinese_label = self.image_files[idx][2]
        return {'img':img, 'label':label, 'chinese_label':chinese_label}

    def __len__(self):
        return len(self.image_files)



if __name__ == "__main__":
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    traindataset= ImageDataset("./food_data/", "./food_data/label2name.txt", transforms_, mode="train", image_type="r")
    train_dataloader = DataLoader(traindataset, batch_size=2)
    for data in train_dataloader:
        import pdb; pdb.set_trace()
