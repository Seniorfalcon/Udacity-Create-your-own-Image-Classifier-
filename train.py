import numpy as np

import torch
from torch import nn
from torch import tensor
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import train_functions

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="flowers")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint_2.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)
ap.add_argument('--output_units', type=int, dest="output_units", action="store", default=4096)



pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_size = pa.hidden_units
out_size = pa.output_units

device = pa.gpu
epochs = pa.epochs

def main():
    
    trainloader, validloader, testloader = train_functions.load_data(root)
    model, optimizer, criterion = train_functions.Network(structure, dropout, hidden_size, out_size, lr)
    train_functions.train_model(model, optimizer, criterion, epochs, t_loader = trainloader, v_loader =validloader,  device ='gpu')
    train_functions.save_checkpoint(model, path, structure, hidden_size, out_size, dropout, lr)
    print("Training has been done ...!")

if __name__== "__main__":
    main()