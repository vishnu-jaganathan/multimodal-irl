import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

class AtariImageDataset(Dataset):
    def __init__(self, image_dir):
        self.img_dir = image_dir
        self.image_names = pd.read_csv(os.path.join(image_dir,'image_path.csv'))

    def __len__(self):
        return len(self.image_names) - 1

    def __getitem__(self, time):
        img1_path = self.image_names.iloc[time,1]
        img2_path = self.image_names.iloc[time+1,1]
        image1 = read_image(img1_path).to(torch.float)
        image2 = read_image(img2_path).to(torch.float)
        
        image = torch.stack([image1[0],image2[0]])
        return image, image