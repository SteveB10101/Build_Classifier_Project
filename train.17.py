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



# Argument parsing section
# set up default values for arguments that will be parsed

default_learn_rate = 0.007
default_data_dir = "flowers"
default_architecture = "densenet121"
default_num_epochs = 5
default_hidden_units = [16384]
default_checkpoint = "zzz.pth"

ap = argparse.ArgumentParser()

'''
ap.add_argument("-n", "--name", required=True,
	help="name of the user")
'''

ap.add_argument("-lr", "--learning_rate", default = default_learn_rate, type = float,
help = "Enter learning rate less than or equal to 1.0 ")

ap.add_argument('-dir', '--data_dir', default = default_data_dir, type = str,
help = "Enter path to data directory")

ap.add_argument('-arch', '--architecture', default =default_architecture, type = str,
help = "Choose any architecture, as long as it is {}".format(default_architecture))

ap.add_argument('-ne', '--num_epochs', default = default_num_epochs, type = int,
help ="Enter the number of desired Epochs")

ap.add_argument('-hu', '--hidden_units', default = default_hidden_units, type = int,
help = "Choose the number of hidden units")

ap.add_argument('-cp', '--checkpoint', default =default_checkpoint, type = str,
help = "Save the trained model checkpoint to disk")

ap.add_argument('--gpu', action = 'store_true',
help = "Choose GPU if available, else CPU")

#args = vars(ap.parse_args())
args = ap.parse_args()
print(args)


print(args.learning_rate)
#if args['learning_rate'] > 1.0:
if args.learning_rate > 1.0:
	print("The learning rate of {} is greater than 1.0 ".format(args.learning_rate))
	args.learning_rate = default_learn_rate
	print("Reverting to default learning rate of {}".format(args.learning_rate))


# display a friendly message to the user
#print("Hi there {}, it's nice to meet you!".format(args["name"]))
print("The learning rate is set to {}".format(args.learning_rate))
#print(args.learning_rate)
print("The data directory is set to {}".format(args.data_dir))
print("The chosen architecture is {}".format(args.architecture))
print("The number of epochs is set to {}".format(args.num_epochs))
print("The number of hidden units is set to {}".format(args.hidden_units))
print("The model will be saved as {}".format(args.checkpoint))
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
'''
u have to modify your original code such that it recognizes the parsed command line arguments.
e.g. if you parsed your command line arguments in a variable args and the command option for epochs was epoch,
you will get the value in args.epoch.
So, you will use something like epochs = args.epoch.

'''


'''
BUILDING AND TRAINING THE CLASSIFIER

At this point using the pretrained Densenet121 architecture
'''

# TODO: Build and train your network
#Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)

model_densenet121 = models.densenet121(pretrained = True)

# Freeze parameters so we don't backpropagate through them
for param in model_densenet121.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout


# Hyperparameters for network

input_size = 1024
output_size = 102 # This is the number of flower categories
hidden_layers = args.hidden_units #[16384]
drop_p = 0.2

print('input size ' , input_size)
print('output_size ', output_size)
print('hidden_layers ', hidden_layers)
print('dropout ', drop_p)

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

# Create the network, define the criterion and the optimizer
print('Hyperparameters:')
print('input_size:  ', input_size, '  output_size:  ', output_size)
print('hidden layers:  ', hidden_layers)
print('drop probability  ', drop_p)

model = Network(input_size, output_size, hidden_layers, drop_p )

#model


# Assign the classifier attribute of model_densenet
model_densenet121.classifier = model

# IF GPU is enabled from the command line, then it will be enabled IF available
if args.gpu:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else: #ELSE occurs if GPU is NOT enabled from the command line
	device = torch.device("cpu")

input = model.to(device)
print('device ', device)
criterion = nn.NLLLoss()

#learn_rate = 0.007
#the learning_rate now comes from command line input, or the default learning rate
learn_rate = args.learning_rate


#We are interested in transfer learning by only training the final classifier,
#So we need to optimize only the classifier

optimizer = optim.Adam(model_densenet121.classifier.parameters(), lr = learn_rate)

print('learning rate  ', learn_rate)

params_dict = {'input_size' : input_size, 'output_size' : output_size,
               'hidden_layers' : hidden_layers, 'drop probability' : drop_p,
               'learning_rate' : learn_rate}

print(params_dict)


#TESTING THE NETWORK


# TODO: Do validation on the test set
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        #turn this on later when running GPU
        images, labels = images.to(device), labels.to(device)

        #output = model.forward(images)
        #Edit: use the densenet model
        ##output = model_densenet121.forward(images)
        ##Edit again: use model
        output = model.forward(images)

        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


# Keep session active
# DOESNT WORK!!!!!!!!!!!!!!
#with active_session:
# Run validation

###################
#model_densenet121.to('cuda')
model_densenet121.to(device)
# number of epochs now comes from command line, or the default value
epochs = args.num_epochs
steps = 0
running_loss = 0
print_every = 51 #40
count = 0

# update parameters dictionary with epochs:
params_dict['num_epochs'] = epochs

print(params_dict)

for e in range(epochs):
    if count > 0:
        print('epoch=', e, 'count= ', count-1)
    count = 0
    ## Edit 1: change model.train() to model_densenet121.train()
    # model.train()
    model_densenet121.train()
    for images, labels in trainloader:

        images, labels = images.to(device), labels.to(device)

        steps += 1
        count += 1
        #print('steps = ', steps)


        optimizer.zero_grad()

        #output = model.forward(images)

        #Edit: use the densenet model
        output = model_densenet121.forward(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print('made it past optimizer.step')

        if steps % print_every == 0:
            start = time.time()
            # Put Model in eval mode for inference
            ## Edit 2: change model.eval() to model_densenet121.eval()
            #model.eval()
            model_densenet121.eval()

            # Turn off gradients for validation (saves memory and computations)
            with torch.no_grad():
                #test_loss, accuracy = validation(model, testloader, criterion)
                test_loss, accuracy = validation(model_densenet121, testloader, criterion)
            print("Epoch: {}/{}..".format(e+1, epochs),
                "Training Loss: {:.3f}..".format(running_loss/print_every),
                "Test Loss: {:.3f}..".format(test_loss/len(testloader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
                "Steps = ", steps)

            running_loss = 0

            #turn training back on
            ## Edit 3: change model.train() to model_densenet.train()
            #model.train()
            model_densenet121.train()

            print(f"Device = {device}; Time per {print_every} steps: {(time.time() - start)/3:.3f} seconds")


# TODO: Save the checkpoint

#model_densenet121.class_to_idx = image_datasets['train'].class_to_idx
model_densenet121.class_to_idx = train_data.class_to_idx

checkpoint= {'input_size': input_size,
             'output_size' : output_size,
            'hidden_layers' : hidden_layers,
             'num_epochs' : epochs,
             'class_to_idx' : model_densenet121.class_to_idx,
            'state_dict' : model_densenet121.state_dict()}
torch.save(checkpoint, args.checkpoint)
#model_densenet121.class_to_idx
