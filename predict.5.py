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
import json


# Argument parsing section
# set up default values for arguments that will be parsed

default_learn_rate = 0.007
default_data_dir = "flowers"
default_checkpoint_to_load = "yyy.pth"
default_architecture = "densenet121"
default_num_epochs = 5
default_hidden_units = [16384]
#default_checkpoint = "zzz.pth"

ap = argparse.ArgumentParser()

'''
ap.add_argument("-n", "--name", required=True,
	help="name of the user")
'''

ap.add_argument('-dir', '--data_dir', default = default_data_dir, type = str,
help = "Enter path to data directory")

ap.add_argument('-cpload', '--checkpoint_to_load', default = default_checkpoint_to_load, type = str,
help = "Enter path and name of the checkpoint")

#ap.add_argument('-arch', '--architecture', default =default_architecture, type = str,
#help = "Choose any architecture, as long as it is {}".format(default_architecture))

ap.add_argument('-arch', '--architecture', default =default_architecture, type = str,
#help = "Choose any architecture, as long as it is {}".format(default_architecture))
help = "Choose one architecture, either densenet121 or resnet50")

#ap.add_argument('-cp', '--checkpoint', default =default_checkpoint, type = str,
#help = "Save the trained model checkpoint to disk")

ap.add_argument('--gpu', action = 'store_true',
help = "Choose GPU if available, else CPU")

#args = vars(ap.parse_args())
args = ap.parse_args()
print(args)


# display a friendly message to the user
print("The data directory is set to {}".format(args.data_dir))
print("The chosen architecture is {}".format(args.architecture))
#print("The model will be saved as {}".format(args.checkpoint))
print("Checkpoint to be loaded: {}".format(args.checkpoint_to_load))

if args.gpu:
	print("GPU is enabled and will be used if available")
else:
	print("GPU is not enabled")
#

# End argument parsing section
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
    print('inside load_checkpoint function')
    print('filepath: ', filepath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # MODEL WAS TRAINED ON GPU, need to switch to CPU if necessary!
    # Hardcoded 'cpu' below
    checkpoint = torch.load(filepath, map_location='cpu')

    print('past checkpoint = torch.load(filepath)')

	#Load a pre-trained network
    if args.architecture == 'densenet121':
        model_chosen_arch = models.densenet121(pretrained = True)
        input_size = 1024
    elif args.architecture == 'resnet50':
        model_chosen_arch = models.resnet50(pretrained = True)
        input_size = 2048
    else:
        print('Architecture Choice Not Valid')
        print(args.architecture)


    #model_chosen_arch = models.densenet121(pretrained=True)


    model = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'])

    model_chosen_arch.classifier = model

    model_chosen_arch.load_state_dict(checkpoint['state_dict'])

    model_chosen_arch.class_to_idx = checkpoint['class_to_idx']

    return model_chosen_arch

'''
IMAGE PREPROCESSING

'''

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model

    # Step 1: Resize so that the short size is 256 pixels, retain aspect ratio
    short_side = 256
    #print('short side ', short_side)
    im = Image.open(image)
    print('image open')
    print('image height, width:  ', im.height, im.width)

    aspect_ratio = im.height / im.width
    print('aspect_ratio: ', aspect_ratio)

    if im.height >= im.width:
        resize_width = short_side
        #resize_height = int(im.height * short_side / im.width)
        resize_height = int(aspect_ratio * im.height)
    else:
        resize_height = short_side
        #resize_width = int(im.width * short_side / im.height)
        resize_width = int(aspect_ratio * im.width)

    print('im_height: ', im.height, 'im_width: ', im.width, 'resized height: ', resize_height, 'resized width: ', resize_width)

    # resized image
    #im_resized = im.resize((resize_height, resize_width))
    im_resized = im.resize((resize_width, resize_height))

    # Step 2: Crop out 224 x 224 from center

    crop_length = 224
    left = (resize_width - crop_length) / 2
    upper = (resize_height - crop_length) / 2
    right = left + crop_length #resize_width
    lower = upper + crop_length #resize_height
    # left, upper, right, lower
    print('crop coords: ', left, upper, right, lower, '\n')
    im_cropped = im_resized.crop((left, upper, right, lower))

    #normalize across pixels:
    np_image = np.array(im_cropped)/255

    #print('shape of image after cropping and converting to numpy ', np_image.shape)

    # convert color channels to values from 0 to 1

    means = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])

    #print('np_image before transpose: ', np_image.shape)

    #Normalize the image
    #subtract the means, then divide by standard deviation
    # do this operation on the color channel (currently the first channel)

    np_image = (np_image - means) / stdev

    #transpose image. Move first dimension to 3rd dimension
    # color channel is 3rd dimension in PIL and Numpy images. Pytorch expects it to be in 1st dimension

    np_image = (np.transpose(np_image, (2, 0, 1)))

    #print('process_image function output: ')
    #print('np_image after transpose: ', np_image.shape)
    #print('np_image type: ', type(np_image), '\n')

    return np_image

# From Part 1:
# To check your work, the function below converts a PyTorch tensor
# and displays it in the notebook.
# If your process_image function works, running the output through this function
# should return the original image (except for the cropped out portions).

# imshow function NOT USED
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""

    print('inside imshow function 1.0')

    if ax is None:
        fig, ax = plt.subplots()

    #print('Inside imshow function, image shape, type: ', image.shape, type(image))

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension

    #
    # This function supposedly is expecting a Pytorch tensor and converts it to a Numpy array here?
    # Extra unnecessary steps.
    print('inside imshow function 2.0')
    image = image.numpy().transpose((1, 2, 0))
    print('inside imshow function 3.0')
    #print('Inside imshow function after numpy conversion and transpose', '\n')
    #print('image shape, type: ', image.shape, type(image))

    #image = (np.transpose(image, (1, 2, 0)))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    print('inside imshow function 4.0')
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    print('Inside imshow function 5.0')

    ax.imshow(image)

    return ax

##############################################D#####
# Grab an image, prepocess it, (then DON'T show the image):
###################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Find out where image files are relative to workspace

cwd = os.getcwd()
print(cwd)


## not needed currently, but for listing images in folder:
#files = os.listdir(os.curdir + '/flowers/train/1')
#print(files)

#data_dir = 'flowers'
train_dir = args.data_dir + '/train'


# grab an image from 'train_dir'

#test_image = (train_dir + '/1/image_06761.jpg')
#test_image = ('../testing/IMG_6791.JPG')
test_image = (train_dir + '/12/image_04024.jpg')
# process_image function returns a Numpy array
processed_image = process_image(test_image)
print('Returned from process_image function')

# The project says that the input for the imshow function is a PyTorch Tensor

processed_image_t = torch.from_numpy(processed_image)

#print('\n', 'image size and type before inputting to imshow function  ', processed_image_t.size(), type(processed_image_t), '\n')

# NOT REQUIRED to display image

#imshow(processed_image_t)

########################
# CLASS PREDICTION
########################

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file

    # print('Inside Predict Function')

    # From Part 5 Inference and Validation
    model.eval()


    # process_image returns a numpy array

    image = process_image(image_path)
    # print('still inside predict function, returned from process_image function')
    # print('image type ', type(image))

    # convert image from numpy array to pytorch tensor

    image = torch.from_numpy(np.array([image]))

    # print('image type before image.float, should be tensor', type(image))

    image = image.float()


    #print('still inside predict function, converted image from numpy array to pytorch tensor')
    #print('image type ', type(image))

    with torch.no_grad():
        log_probs = model.forward(image)

    #exponential function required because of softmax in last layer
    linear_probs = torch.exp(log_probs)

    topk_probs, topk_indices = linear_probs.topk(topk)

    # print(topk_probs)

    # print(topk_indices)

    # print(model.class_to_idx)

    # Invert dictionary (model.class_to_idx) from load_checkpoint function to get mapping from index to class
    # model_chosen_arch.class_to_idx = checkpoint['class_to_idx']

    idx_to_class ={}

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # print('inverted dictionary: ', idx_to_class)

    # Convert topk_indices to a list to iterate through to get the classes
    #topk_indices_list = np.array(topk_indices)

    topk_indices_list = topk_indices[0].tolist()

    # convert topk probabilities from a tensor to a list- apparently the next step(s) will require a list
    topk_probs_list = topk_probs[0].tolist()

    print('\n topk_indices_list: ', topk_indices_list, '\n')
    print('\n topk_probs_list: ', topk_probs_list, '\n')

    #class_from_idx_list = []

    #for index in topk_indices_list:
    #    print(index, [idx_to_class[index]])
        #class_from_idx_list += [idx_to_class[index]]

    topk_classes = [idx_to_class[index] for index in topk_indices_list]

    return topk_probs_list, topk_classes #

# Load the model from the checkpoint using the load_checkpoint function
print('device: ', device)

### is the following necessary???
# model_chosen_archnet121.to(device)

cwd = os.getcwd()
print(cwd)

# for listing files in current working directory:
# need to find where the checkpoint was saved. Or else RETRAIN AND SAVE FOR THE 3rd TIME!!!!!!!!!!!!!!!!!!!!!!!
#files = os.listdir(os.curdir)
#print(files)

model = load_checkpoint(args.checkpoint_to_load)

#Use predict function here

#image_path = train_dir + '/1/image_06735.jpg'
#image_path = train_dir + '/12/image_04024.jpg'
image_path = '../testing/IMG_6791.JPG'
#image_path = 'IMG_6791.JPG'
probs, classes = predict(image_path, model)

print('probs:',  probs)
print('classes: ', classes)

# use cat to names to convert numerical classes to names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

names = [cat_to_name[i] for i in classes]

print(names)
