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


def evaluate(model, loader, max_batches=None):
    model.eval()
    with torch.no_grad():
        pos_f1 = BinaryF1Score()
        neg_f1 = BinaryF1Score()
        for images1, images2, label in islice(tqdm(loader, desc='Evaluating model'), max_batches):
            distance = model.forward(images1.to(model.device), images2.to(model.device)).cpu()
            pos_f1.update(distance < model.threshold, label)
            neg_f1.update(distance > model.threshold, 1 - label)
    return (pos_f1.compute() + neg_f1.compute()) / 2


def train(model, train_loader, train_threshold_loader, valid_loader=None, test_loader=None, epochs=20, lr=1e-4, max_batches=None, print_fscore=False):
    print("Debug: Initializing ContrastiveLoss and Optimizer")
    criterion = ContrastiveLoss().to(model.device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(epochs):
        for images1, images2, label in islice(tqdm(train_loader, desc='Training model'), max_batches):
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
        
        print(f'Epoch {epoch} | Loss:{loss.item()}')
        if print_fscore:
            print(f'Train fscore: {evaluate(model, train_loader, max_batches=max_batches)}')
            print(f'Valid fscore: {evaluate(model, valid_loader, max_batches=max_batches)}')
    if print_fscore:
        print(f'Test fscore: {evaluate(model, test_loader, max_batches=max_batches)}')