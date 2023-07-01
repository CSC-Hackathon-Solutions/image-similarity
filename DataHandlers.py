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
    def __init__(self, data, path='data/images', transform=None):
        self.data = data
        self.transform = transform
        self.path = path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        name1 = sample['Name1']
        name2 = sample['Name2']

        image1 = Image.open(os.path.join(self.path, name1))
        image2 = Image.open(os.path.join(self.path, name2))

        # Convert images to RGB if they are not
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        if 'Equal' in sample:    
            equal = sample['Equal']
            equal = torch.tensor(equal)
            return image1, image2, equal
        elif 'ID' in sample:    
            id_ = sample['ID']
            id_ = torch.tensor(id_)
            return image1, image2, id_
        else:
            assert(false)


    

class InMemDataLoader(object):
    """
    A data loader that keeps all data in CPU or GPU memory.
    """

    __initialized = False

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        drop_last=False,
        augment=False,
    ):
        """A torch dataloader that fetches data from memory."""
        batches = []
        for i in tqdm(range(len(dataset))):
            batch = [t.clone().detach() for t in dataset[i]]
            batches.append(batch)
        tensors = [torch.stack(ts) for ts in zip(*batches)]
        dataset = torch.utils.data.TensorDataset(*tensors)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.augment = augment

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last
            )

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ("batch_size", "sampler", "drop_last"):
            raise ValueError(
                "{} attribute should not be set after {} is "
                "initialized".format(attr, self.__class__.__name__)
            )

        super(InMemDataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        for batch_indices in self.batch_sampler:
            batch = self.dataset[batch_indices]
            if self.augment:
                xs = []
                ys = []
                ts = T.Compose([
                    T.ToPILImage(),
                    T.RandomRotation(10),
                    T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1)),
                    T.ToTensor(),
                ])
                num_samples = 2
                for x, y in zip(batch[0], batch[1]):
                    for _ in range(num_samples):
                        xs.append(ts(x))
                        ys.append(y)
                yield(torch.stack(xs), torch.stack(ys))
            else:
                yield batch


    def __len__(self):
        return len(self.batch_sampler)

    def to(self, device):
        self.dataset.tensors = tuple(t.to(device) for t in self.dataset.tensors)
        return self