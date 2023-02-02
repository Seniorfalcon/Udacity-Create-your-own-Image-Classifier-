import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import json
import PIL
from PIL import Image
import argparse

import predict_fun

ap = argparse.ArgumentParser(description='Predict.py')

ap.add_argument('input', default='./flowers/test/10/image_07090.jpg', nargs='?', action="store", type = str)
ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
ap.add_argument('checkpoint', default='./checkpoint_2.pth', nargs='?', action="store", type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
path_image = pa.input
number_of_outputs = pa.top_k
device = pa.gpu
categ = pa.category_names
path = pa.checkpoint


def main():
    model=predict_fun.load_checkpoint(path)
    with open(categ , 'r') as f:
        cat_to_name = json.load(f, strict=False)
        
    probabilities, classes = predict_fun.predict(path_image, model, number_of_outputs, device)
    #probabilities = predict_fun.predict(path_image, model, number_of_outputs, device)
    #labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    class_names = [cat_to_name[item] for item in classes]
    
    #probability = np.array(probabilities[0][0])
    i=0
    while i < number_of_outputs:
        print("{} with a probability of {:.3f}".format(class_names[i], probabilities[i]))
        i += 1
    print("...")    
    print("Prediction has been done!")

    
if __name__== "__main__":
    main()