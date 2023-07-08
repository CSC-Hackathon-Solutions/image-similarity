"""
    Imports
"""

import requests
from io import BytesIO
from PIL import Image
import os
from itertools import islice
from typing import Literal

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
from torchmetrics.classification import BinaryF1Score, BinaryConfusionMatrix

from torchvision.models import resnet152
from torchvision.models.resnet import ResNet152_Weights

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import optuna

def tqdm_loader(loader, desc=None, max_batches=None):
    if max_batches is None:
        total = len(loader)
    else:
        total = min(max_batches, len(loader))
    return tqdm(islice(loader, max_batches), desc=desc, total=total)


"""
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec for more details 
"""
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean(label * torch.pow(distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0), 2))

        return loss_contrastive


def evaluate(model, loader, max_batches=None, show_progress_bar=True):
    model.eval()
    with torch.no_grad():
        pos_f1 = BinaryF1Score()
        neg_f1 = BinaryF1Score()
        if show_progress_bar:
            loader = tqdm_loader(loader, max_batches=max_batches, desc='Evaluating')
        for images1, images2, label in loader:
            distance = model.forward(images1.to(model.device), images2.to(model.device)).cpu()
            pos_f1.update(distance < model.threshold, label)
            neg_f1.update(distance > model.threshold, 1 - label)
    return (pos_f1.compute() + neg_f1.compute()) / 2


def train(model, train_loader, train_threshold_loader, valid_loader=None, test_loader=None, optimizer=None, epochs=20, lr=1e-4, max_batches=None, verbose=False, show_progress_bar=True):
    criterion = ContrastiveLoss().to(model.device)
    if optimizer is None:
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    for epoch in range(epochs):
        if show_progress_bar:
            # TODO why we have to take max_batches - 1 here ??? 
            # Denys Zinoviev
            loader = tqdm_loader(train_loader, max_batches=max_batches, desc=f'Epoch {epoch}')
        else:
            loader = train_loader
        for images1, images2, label in loader:
            model.train()
            images1 = images1.to(model.device)
            images2 = images2.to(model.device)
            label = label.to(model.device)

            optimizer.zero_grad()
            outputs = model.forward(images1, images2)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        model.update_threshold(train_threshold_loader, max_batches=max_batches)
        
        if verbose:
            print(f'Epoch         : {epoch}')
            print(f'Loss          : {loss.item():.6f}')
            print(f'Train fscore  : {evaluate(model, train_loader, max_batches=max_batches):.4f}')
            print(f'Valid fscore  : {evaluate(model, valid_loader, max_batches=max_batches):.4f}')
    if verbose:
        print(f'Test fscore   : {evaluate(model, test_loader, max_batches=max_batches):.4f}')


def calc_confusion_matrix(model, loader, max_batches=None):
    model.eval()
    with torch.no_grad():
        result = BinaryConfusionMatrix()
        for images1, images2, label in tqdm_loader(loader, max_batches=max_batches, 
                                                   desc=f'Computing confusion matrix'):
            result.update(model.predict(images1.to(model.device), images2.to(model.device)).cpu(), label)
    return result.compute()  


def convert_for_imshow(x):
    return (x.permute(1, 2, 0) + 1) / 2


def split_dataframe(df, ratio, shuffle=True):
    assert(sum(ratio) == 1)
    if shuffle:
        df = df.sample(frac=1)
    return np.split(df, (np.cumsum(ratio[:-1]) * df.shape[0]).astype(int))

"""
    Test whether loaders are working.
"""
def test_loader_images(loader):
    print('Examples of images from loader:')
    image1, image2, _ = next(iter(loader))
    image1 = image1[0]
    image2 = image2[0]
    _, axs = plt.subplots(1, 2, figsize=(15, 15))

    axs[0].imshow(convert_for_imshow(image1))
    axs[0].set_title('Image 1')
    axs[0].axis('off')
    axs[1].imshow(convert_for_imshow(image2))
    axs[1].set_title('Image 2')
    axs[1].axis('off')
    plt.show()

        
"""
    Tries to find and show mislabeled images from the specified loader.
"""
def mislabeled(model, loader, mislabeled_type: Literal['pred_true', 'pred_false', 'all'] = 'all'):
    def mislabeled_inner():
        for images1, images2, equal in loader:
            preds = model.predict(images1, images2)
            if mislabeled_type:
                if mislabeled_type == 'pred_true':
                    mask = (preds == 1) & (equal == 0)
                else:
                    mask = (preds == 0) & (equal == 1)
            else:
                mask = preds != equal
            for index in mask.nonzero().reshape(-1).tolist():
                yield (images1[index], images2[index], preds[index], equal[index])

    button = widgets.Button(description="Next Images")
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            clear_output()
            image1, image2, pred, truth = next(mislabeled_gen)
            fig, axs = plt.subplots(1, 3, figsize=(16,4))

            axs[0].imshow(convert_for_imshow(image1))
            axs[0].set_title('Image 1')
            axs[1].imshow(convert_for_imshow(image2))
            axs[1].set_title('Image 2')
            axs[2].imshow(convert_for_imshow(torch.abs(image1 - image2)))
            axs[2].set_title('Delta')

            suptitle = f'Predicted: {pred.item()}\nTruth: {truth.item()}'
            fig.suptitle(suptitle)
            plt.show()

    button.on_click(on_button_clicked)
    display(button, output)
    mislabeled_gen = mislabeled_inner()
    on_button_clicked(None)


"""
    Saves predictions that should be submitted to kaggle.
"""
def save_submission(model, loader, max_submit_id, path='data/submission.csv'):
    print(f'Started saving test predictions to {path}')
    ids = []
    preds = []
        
    for images1, images2, id_ in tqdm_loader(loader, desc='Saving submission predictions'):
        preds.extend(model.predict(images1, images2))
        ids.extend(id_)
        
    # all_ids = pd.DataFrame({
    #     'ID': range(2, max_submit_id + 1),
    # })
    res = pd.DataFrame({
        'ID': [obj.item() for obj in ids],
        'same': [obj.item() for obj in preds],
        'different': [1 - obj.item() for obj in preds]
    }).drop_duplicates()

    # res = all_ids.merge(res, on='ID', how='left').fillna(0)
    res.to_csv(path, index=False)
    print(f'Saved test predictions to {path}\n')


"""
    Returns indices of example dataframe in decreasing order of example difficulty
"""
def sort_example_hardness(df, model, threshold, transform=None, batch_size=32, max_batches=None):
    with torch.no_grad():
        if max_batches is not None:
            df = df[:max_batches * batch_size]
        loader = DataLoader(ImageDataset(df, transform=transform), batch_size=batch_size, shuffle=False)
        differences = []
        for images1, images2, _ in tqdm_loader(loader, max_batches=max_batches, desc='Sort example hardness'):
            distance = model.forward(images1.to(model.device), images2.to(model.device)).cpu()
            differences.append(torch.abs(threshold - distance))
        differences = torch.cat(differences)
        return torch.argsort(differences)

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
            assert(False)


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