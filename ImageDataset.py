import requests
from io import BytesIO
from PIL import Image
import os

from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, data, path='data/', transform=None):
        self.data = data
        self.transform = transform
        self.path = path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path1 = sample['Path1']
        path2 = sample['Path2']
        equal = sample['Equal']
        
        image1 = Image.open(self.path + path1)
        image2 = Image.open(self.path + path2)
        
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return image1, image2, equal
