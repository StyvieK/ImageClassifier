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
from utils import load_model
from utils import predict
from utils import process_image

parser = argparse.ArgumentParser(
    description='Predict Image classisifer',
)

parser.add_argument('imagepath', action="store" , help='Folder containing training images')
parser.add_argument('checkpoint', action="store")
parser.add_argument('--GPU', action="store_true", dest="GPU", default=False,help='Use GPU if available')
parser.add_argument('--top_k', action="store", dest="top_k", default=1, type=int,help='K classes')
parser.add_argument('--category_names', action="store", dest="category_names", help='Parse category Name')

results = parser.parse_args()
if(True):
    print('Running with params')
    print('imagepath       = {!r}'.format(results.imagepath))
    print('loadcheckpoint  = {!r}'.format(results.checkpoint))
    print('GPU             = {!r}'.format(results.GPU))
    print('Top K           = {!r}'.format(results.top_k))
    print('category_names  = {!r}'.format(results.category_names))
    
model = load_model(results.checkpoint)
print("Loaded model")

#Run on GPU if selected
if results.GPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

if results.GPU and device == "cpu":
    print("No CUDA device found for GPU processing - reverting to CPU")

model.to(device)

indextoclass = {v: k for k, v in model.class_to_idx.items()}                    
                    
probs, classes = predict(results.imagepath, model,device,results.top_k)
    
probs = probs.cpu().numpy().tolist()[0]
classes = classes.cpu().numpy().tolist()[0]

if results.category_names is not None:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f) 
    names = [cat_to_name[indextoclass[item]] for item in classes]
    print("Ranked Names : {}".format(names))
    print("Probabilities: {}".format(probs))   
else:   
    print("Class names not provieded.")
    mappedclass = [indextoclass[item] for item in classes]
    print("Ranked Classes : {}".format(mappedclass))
    print("Probabilities: {}".format(probs))    
