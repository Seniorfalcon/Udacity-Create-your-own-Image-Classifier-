import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import json
import PIL
from PIL import Image
import argparse
import train_functions


def load_checkpoint(path='checkpoint_2.pth'):
    checkpoint = torch.load(path)
    lr=checkpoint['lr']
    hidden_size = checkpoint['hidden_size']
    out_size = checkpoint['out_size']
    dropout = checkpoint['dropout']
    structure = checkpoint['structure']

    model, criterion, optimizer = train_functions.Network(structure , dropout,hidden_size, out_size, lr)

    model.class_to_idx = checkpoint['mapping_class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path='/home/workspace/aipnd-project/flowers/test/10/image_07090.jpg'):


    proc_img = Image.open(image_path)

    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    pymodel_img = prepoceess_img(proc_img)
    return pymodel_img


def predict(image='/home/workspace/ImageClassifier/flowers/test/10/image_07090.jpg', model=0, topk=5,device='gpu'):

    img_torch = process_image(image)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
            ps = torch.exp(output) 
            top_ps, top_classes = ps.topk(topk, dim = 1)

            # converting to list
            probs = top_ps.tolist()[0]
            indxs = top_classes.tolist()[0]

            idx_to_class = {value:key for key, value in model.class_to_idx.items()}
            classes = [idx_to_class.get(x) for x in indxs]
            classes = np.array(classes)
            top_ps = np.array(top_ps)
            top_ps = top_ps.reshape(5,)
            
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
            ps = torch.exp(output) 
            top_ps, top_classes = ps.topk(topk, dim = 1)

            # converting to list
            probs = top_ps.tolist()[0]
            indxs = top_classes.tolist()[0]

            idx_to_class = {value:key for key, value in model.class_to_idx.items()}
            classes = [idx_to_class.get(x) for x in indxs]
            classes = np.array(classes)
            top_ps = np.array(top_ps)
            top_ps = top_ps.reshape(5,)   #to be the same shape on classes :
    
    return top_ps, classes     

