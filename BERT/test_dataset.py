import glob
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, csv_file, transforms_=None):

        """
        root: data directory with testing images 
        transform_: transformation.
	csv_file, sample_submission csv file
        """

        self.transform = transforms.Compose(transforms_)
        self.image_files = [] 

        # retrive the image labels from freq, common or rare.
        df = pd.read_csv(csv_file, dtype=str)
        self.image_ids = df['image_id'].tolist()

        for image_id in self.image_ids:
            self.image_files.append(os.path.join(root, image_id + '.jpg'))

    def __getitem__(self, idx):

        img = Image.open(self.image_files[idx])
        img = self.transform(img)
        return self.image_ids[idx], img 

    def __len__(self):
        return len(self.image_files)



if __name__ == "__main__":
    transforms_ = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    traindataset= ImageDataset("../food_data/test", "../food_data/testcase/sample_submission_comm_track.csv", transforms_)
    train_dataloader = DataLoader(traindataset, batch_size=1)
    for data in train_dataloader:
        import pdb; pdb.set_trace()
