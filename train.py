import argparse

import os
import torch
import json
import numpy as np 
import time
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from utils import get_modelfrom_arch
from utils import model_checkpoint

parser = argparse.ArgumentParser(
    description='Train Image classisifer',
)

parser.add_argument('datadir', action="store" , help='Folder containing training images')
parser.add_argument('--arch', action="store", dest="arch", default="VGG-16",  help='Choose from VGG-13/VGG-16/VGG-19')
parser.add_argument('--save_dir', action="store", dest="save_dir",help='Folder to save checkpoint')
parser.add_argument('--GPU', action="store_true", dest="GPU", default=False,help='Use GPU if available')
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, default=0.001,help='Optimizer learning rate HyperParameter.')
parser.add_argument('--momentum', action="store", dest="momentum", type=float, default=0.9,help='Optimizer momentum HyperParameter')
parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=10,help='Epochs to use in training')


results = parser.parse_args()
if(True):
    print('Running with params')
    print('datadir       = {!r}'.format(results.datadir))
    print('arch          = {!r}'.format(results.arch))
    print('save_dir      = {!r}'.format(results.save_dir))
    print('GPU           = {!r}'.format(results.GPU))
    print('learning_rate = {!r}'.format(results.learning_rate))
    print('momentum      = {!r}'.format(results.momentum))
    print('epochs        = {!r}'.format(results.epochs))

model = get_modelfrom_arch(results.arch)

training = 'train'
validation = 'valid'

data_transforms = {
    training: transforms.Compose([
        transforms.RandomRotation(60),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]),
    ]),
    validation: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(results.datadir, x), 
        transform=data_transforms[x]
    )
    for x in [training, validation]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=64,shuffle=True)
    for x in [training, validation]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [training, validation]}

for x in [training, validation]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))

#Run on GPU if selected
if results.GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

if results.GPU and device == "cpu":
    print("No CUDA device found for GPU processing - reverting to cpu")
   
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(2048, 102),
                                 nn.LogSoftmax(dim=1))

model.classifier = classifier

optimizer = optim.SGD(model.classifier.parameters(), lr=results.learning_rate, momentum=results.momentum)

model.to(device)
criterion = nn.NLLLoss()
criterion.to(device)

model.class_to_idx = image_datasets[training].class_to_idx 

if results.save_dir is None:
    CheckPointFileName = 'ModelCheckpoint.pth'
else:
    CheckPointFileName = os.path.join('.',results.save_dir,'ModelCheckpoint.pth')

for i in range(results.epochs):
    imagestrained = 0
    trainingloss = 0
    for trainingimages, traininglabels in dataloaders[training]:
        
        trainingimages, traininglabels = trainingimages.to(device), traininglabels.to(device)
        
        optimizer.zero_grad()
        
        modeloutput = model.forward(trainingimages)
        loss = criterion(modeloutput, traininglabels)
        loss.backward()
        optimizer.step()
        
        trainingloss += loss.item();
    else:
        validationloss = 0
        accuracy = 0
               
        model.eval()
        with torch.no_grad():
            for validationimages, validationlabels in dataloaders[validation]:
                
                validationimages, validationlabels = validationimages.to(device), validationlabels.to(device)
                logvalidationoutput = model.forward(validationimages)
                currentvalidationloss = criterion(logvalidationoutput,validationlabels)
                    
                validationloss += currentvalidationloss.item()
                    
                validationoutput = torch.exp(logvalidationoutput)
                    
                top_p, top_class = validationoutput.topk(1, dim=1)
                equals = top_class == validationlabels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
               
        print("Epoch: {}/{}.. ".format(i+1, results.epochs),
              "Training Loss: {:.3f}.. ".format(trainingloss/len(dataloaders[training])),
              "Test Loss: {:.3f}.. ".format(validationloss/len(dataloaders[validation])),
              "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders[validation])))
        validationloss = 0
        model.train()     
        
model.class_to_idx = image_datasets[training].class_to_idx            
model_checkpoint(CheckPointFileName,results.epochs,results.arch, model, optimizer,classifier)

print('Done');
