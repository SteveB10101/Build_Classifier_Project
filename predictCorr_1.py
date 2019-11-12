# Imports here

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
import json

# Argument parsing section
# set up default values for arguments that will be parsed

default_data_dir = "flowers"
default_image_path = default_data_dir + '/train/12/image_04024.jpg'
default_cat_to_name = 'cat_to_name.json'
default_checkpoint_to_load = 'orig_checkpoint.pth' #"yyy.pth"
default_architecture = "densenet121"
default_top_k = 1

ap = argparse.ArgumentParser()

ap.add_argument('-dir', '--data_dir', default = default_data_dir, type = str,
help = "Enter path to data directory")

ap.add_argument('-impath', '--image_path', default = default_image_path, type = str,
help = "Enter path to image directory")

ap.add_argument('-name', '--cat_to_name', default = default_cat_to_name, type = str,
help = "Enter path to category to name json file")

ap.add_argument('-top', '--top_k', default = default_top_k, type = int,
help = "Top K most likely classes")

ap.add_argument('-cpload', '--checkpoint_to_load', default = default_checkpoint_to_load, type = str,
help = "Enter path and name of the checkpoint")

# Correction 3- remove this- should be contained in the checkpoint
#ap.add_argument('-arch', '--architecture', default =default_architecture, type = str,
#help = "Choose one architecture, either densenet121 or resnet50")

ap.add_argument('--gpu', action = 'store_true',
help = "Choose GPU if available, else CPU")

args = ap.parse_args()
#print(args)

# display a friendly message to the user
print("The data directory is set to {}".format(args.data_dir))
print("The chosen architecture is {}".format(args.architecture))
print("Checkpoint to be loaded: {}".format(args.checkpoint_to_load))

if args.gpu:
	print("GPU is enabled and will be used if available")
else:
	print("GPU is not enabled")

# End argument parsing section########################################
#
# IMPORT load_data file
import load_data_10

#Load data function- returns trainloader, validloader, dataloader
trainloader, validloader, testloader = load_data_10.load_data(args.data_dir)
#

###################### Insert Network Class Here. ####################

class Network(nn.Module):
    '''
    Using the example from Part 5, Inference and Validation:

    Builds a feedforward network with arbitrary hidden layers.

    Arguments:

    input_size: integer type, size of the input
    output_size: integer type, size of the output layer
    hidden_layers: List type integers, the sizes of the hidden layers
    drop_p: float type between 0.0 and 1.0, dropout probability

    '''
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5):

        super().__init__()
        #Add the fist layer, and input to the first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add arbitrary additional hidden layers:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p = drop_p)

    def forward(self, x):
        # Forward pass through the network, returns the output logits

        # Forward through each layer in hidden_layers, with ReLU activation and dropout

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)
        return F.log_softmax(x, dim = 1)


#####################################################################


######################################
'''
LOADING THE CHECKPOINT
'''
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL WAS TRAINED ON GPU, need to switch to CPU if necessary!
    # Hardcoded 'cpu' below
    checkpoint = torch.load(filepath, map_location='cpu')

    print("architecture from checkpoint: ", checkpoint['architecture'])
    
	#Load a pre-trained network
    if checkpoint['architecture'] == 'densenet121': #args.architecture == 'densenet121':
        model_chosen_arch = models.densenet121(pretrained = True)
        input_size = 1024
    elif checkpoint['architecture'] == 'resnet50': #args.architecture == 'resnet50':
        model_chosen_arch = models.resnet50(pretrained = True)
        input_size = 2048
    else:
        print('Architecture Choice Not Valid')
        print(args.architecture)

    model = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])

    # Assign classifier to correct attribute:
    # "classifier" for densenet121
	# "fc" for resnet50

    if checkpoint['architecture'] == 'densenet121': #args.architecture == 'densenet121':
        model_chosen_arch.classifier = model

    elif checkpoint['architecture'] == 'resnet50': #args.architecture == 'resnet50':
        model_chosen_arch.fc = model

    model_chosen_arch.load_state_dict(checkpoint['state_dict'])

    model_chosen_arch.class_to_idx = checkpoint['class_to_idx']

    return model_chosen_arch

#################################
# IMAGE PREPROCESSING
################################

def process_image(image):
    #Scales, crops, and normalizes a PIL image for a PyTorch model,
    #Returns a Numpy array

    # TODO: Process a PIL image for use in a PyTorch model

    # Step 1: Resize so that the short size is 256 pixels, retain aspect ratio
    short_side = 256
    #print('short side ', short_side)
    im = Image.open(image)

    aspect_ratio = im.height / im.width

    if im.height >= im.width:
        resize_width = short_side
        #resize_height = int(im.height * short_side / im.width)
        resize_height = int(aspect_ratio * im.height)
    else:
        resize_height = short_side
        #resize_width = int(im.width * short_side / im.height)
        resize_width = int(aspect_ratio * im.width)

    # resized image

    im_resized = im.resize((resize_width, resize_height))

    # Step 2: Crop out 224 x 224 from center

    crop_length = 224
    left = (resize_width - crop_length) / 2
    upper = (resize_height - crop_length) / 2
    right = left + crop_length #resize_width
    lower = upper + crop_length #resize_height
    # left, upper, right, lower
    #print('crop coords: ', left, upper, right, lower, '\n')
    im_cropped = im_resized.crop((left, upper, right, lower))

    #normalize across pixels:
    np_image = np.array(im_cropped)/255

    # convert color channels to values from 0 to 1

    means = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])

    #Normalize the image
    #subtract the means, then divide by standard deviation
    # do this operation on the color channel (currently the first channel)

    np_image = (np_image - means) / stdev

    #transpose image. Move first dimension to 3rd dimension
    # color channel is 3rd dimension in PIL and Numpy images. Pytorch expects it to be in 1st dimension

    np_image = (np.transpose(np_image, (2, 0, 1)))

    return np_image
# Correction 4:
# IF GPU is enabled from the command line, then it will be enabled IF available
if args.gpu:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: #ELSE occurs if GPU is NOT enabled from the command line
	device = torch.device("cpu")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


########################
# CLASS PREDICTION
########################

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    # From Part 5 Inference and Validation
    model.eval()

    # process_image returns a numpy array

    image = process_image(image_path)

    # convert image from numpy array to pytorch tensor

    image = torch.from_numpy(np.array([image]))

    image = image.float()

    with torch.no_grad():
        log_probs = model.forward(image)

    #exponential function required because of softmax in last layer
    linear_probs = torch.exp(log_probs)

    topk_probs, topk_indices = linear_probs.topk(topk)

    # Invert dictionary (model.class_to_idx) from load_checkpoint function to get mapping from index to class

    idx_to_class ={}

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Convert topk_indices to a list to iterate through to get the classes
    #topk_indices_list = np.array(topk_indices)

    topk_indices_list = topk_indices[0].tolist()

    # convert topk probabilities from a tensor to a list- apparently the next step(s) will require a list
    topk_probs_list = topk_probs[0].tolist()

    #print('\n topk_indices_list: ', topk_indices_list, '\n')
    #print('\n topk_probs_list: ', topk_probs_list, '\n')

    topk_classes = [idx_to_class[index] for index in topk_indices_list]

    return topk_probs_list, topk_classes #

# Load the model from the checkpoint using the load_checkpoint function
print('device: ', device)

cwd = os.getcwd()
print(cwd)

# for listing files in current working directory:
#files = os.listdir(os.curdir)
#print(files)

model = load_checkpoint(args.checkpoint_to_load)

#Use predict function here

image_path = args.image_path
#image_path = 'IMG_6791.JPG'
probs, classes = predict(image_path, model, args.top_k)

print('Top {} probabilities: {}'.format(args.top_k, probs))
print('Corresponding Top {} Classes: {}'.format(args.top_k, classes))

# use json file to convert numerical classes to names

with open(args.cat_to_name, 'r') as f:
    cat_to_name = json.load(f)

names = [cat_to_name[i] for i in classes]
print('Corresponding Top {} Names: {}'.format(args.top_k, names))
