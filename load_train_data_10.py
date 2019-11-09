# Import this file into training file
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models

#import helper
import time

from PIL import Image
import numpy as np

import os

import argparse


#Load data function
def load_data(data_path = 'flowers'):
    data_dir = data_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    print("training, validation and testing transforms defined")


    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir, transform= train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform= val_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= test_transforms)

    print("training validation and testing datasets loaded")

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    print("trainloader, validloader and testloader defined")

    #return trainloader, validloader, testloader
    return train_data

#load_data()
