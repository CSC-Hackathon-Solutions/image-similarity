"""
    Imports
"""

import requests
from io import BytesIO
from PIL import Image
import os
from itertools import islice

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets

import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryF1Score

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

"""
    Helper functions for splitting pandas dataframes and denormalizing tensors.
"""
def split_dataframe(df, ratio, shuffle=True):
    assert(sum(ratio) == 1)
    if shuffle:
        df = df.sample(frac=1)
    return np.split(df, (np.cumsum(ratio[:-1]) * df.shape[0]).astype(int))

def denormalize_tensor(img):
    return (img.permute(1, 2, 0) + 1) / 2

"""
    Test whether loaders are working.
"""
def test_loader_images(loader):
    print('Examples of images from loader:')
    image1, image2, _ = next(iter(loader))
    image1 = image1[0]
    image2 = image2[0]
    _, axs = plt.subplots(1, 2, figsize=(15, 15))

    axs[0].imshow(denormalize_tensor(image1))
    axs[0].set_title('Image 1')
    axs[0].axis('off')
    axs[1].imshow(denormalize_tensor(image2))
    axs[1].set_title('Image 2')
    axs[1].axis('off')
    plt.show()

        
"""
    Tries to find and show mislabeled images from the specified loader.
"""
def mislabeled(model, loader):
    def mislabeled_inner():
        for images1, images2, equal in loader:
            preds = model.predict(images1, images2)
            for index in (preds != equal).nonzero().reshape(-1).tolist():
                yield (images1[index], images2[index], preds[index], equal[index])

    button = widgets.Button(description="Next Images")
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            clear_output()
            image1, image2, pred, truth = next(mislabeled_gen)
            fig, axs = plt.subplots(1, 3, figsize=(14,7))
            axs[0].imshow(denormalize_tensor(image1))
            axs[1].imshow(denormalize_tensor(image2))
            axs[2].imshow(np.abs(denormalize_tensor(image1 - image2)))
            suptitle = f'Predicted: {pred.item()}\nTruth: {truth.item()}'
            # try:
            #     suptitle += f'\nmodel.forward(): {model.forward(image1, image2)}'
            # except:
            #     pass
            fig.suptitle(suptitle)
            plt.show()

    button.on_click(on_button_clicked)
    display(button, output)
    mislabeled_gen = mislabeled_inner()
    on_button_clicked(None)  # show the first images


"""
    Saves predictions that should be submitted to kaggle.
"""
def save_submission(model, loader, max_submit_id, path='res.csv'):
    print(f'Started saving test predictions to {path}')
    ids = []
    preds = []
        
    for images1, images2, id_ in tqdm(loader):
        preds.extend(model.predict(images1, images2))
        ids.extend(id_)
        
    all_ids = pd.DataFrame({
        'ID': range(2, max_submit_id + 1),
    })
    res = pd.DataFrame({
        'ID': [obj.item() for obj in ids],
        'is_same': [obj.item() for obj in preds]
    }).drop_duplicates()

    res = all_ids.merge(res, on='ID', how='left').fillna(0)
    res.to_csv(path, index=False)
    print(f'Saved test predictions to {path}\n')

"""
    Custom Dataset implementation.
"""

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


"""
    Custom loader that keeps all data in CPU or GPU memory.
"""

class InMemDataLoader(object):

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