# TODO: Save the checkpoint 
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

def model_checkpoint(filename, epochs, arch, model, optimizer, classifier):
    checkpoint = {'epochs': epochs,            
                  'arch': arch,
                  'classifier': classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }   
    torch.save(checkpoint, filename)
    
def get_modelfrom_arch(arch):
    if arch == 'VGG-13':
        model  = models.vgg13(pretrained=True)
    elif arch == 'VGG-16':        
        model  = models.vgg16(pretrained=True)
    elif results.arch == 'VGG-19':
        model  = models.vgg19(pretrained=True)
    else:    
        raise ValueError("Invalid arch param . Supported VGG-13/VGG-16/VGG-19" )        
    return model      


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(filepath):
    checkpoint = torch.load(filepath) #Allow reload on CPU. , map_location='cpu'
    model = get_modelfrom_arch(checkpoint['arch'])
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    numberofepochs = checkpoint['epochs']
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    width ,height = img.size
    
    if width < height:
        new_width  = 256
        new_height = int(new_width * height / width)
    else: 
        new_height = 256
        new_width  = int(new_height * width / height)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    sizeneeded = 224
        
    left = (new_width - sizeneeded) / 2
    top = (new_height - sizeneeded) / 2
     
    img = img.crop((left,top,left+sizeneeded,top+sizeneeded))
        
    img = np.array(img)/255 #convert RGB to float 
    img = (img - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225]) #Normalisze
        
    img = np.transpose(img,(2,0,1))
    return img

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        model.eval()
        img = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)     
        img.unsqueeze_(0)
        output = torch.exp(model.forward(img.to(device)))
        top_p, top_class = output.topk(topk, dim=1)    
        return top_p, top_class 
    