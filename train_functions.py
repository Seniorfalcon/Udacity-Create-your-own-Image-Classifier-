import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch import optim
import argparse
import json
from torch.autograd import Variable

# function for loading data :
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform = train_transform)
    test_data = datasets.ImageFolder(test_dir, transform = test_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transform)



    # define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return trainloader, validloader, testloader


model_input_size = {"vgg16":25088,"vgg13":25088,"densenet121":1024,"alexnet":9216}
def Network(structure = 'vgg16', dropout = 0.5, hidden_size = 4096, out_size = 102, lr = 0.001,device = 'gpu'):

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Please try for vgg16 or densenet121 only")

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(model_input_size[structure], hidden_size),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(hidden_size, 1024),
                              nn.ReLU(),
                              nn.Dropout(dropout),
                              nn.Linear(1024,out_size),
                              nn.LogSoftmax(dim = 1))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, epochs = 5, t_loader = 0, v_loader = 0, device ='gpu'):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = epochs
    for e in range(epochs):
        tot_train_loss = 0
        for images, labels in t_loader:
            if torch.cuda.is_available() and device == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            tot_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            
        else:
            tot_valid_loss = 0
            accuracy = 0
            model.eval()
            
            # Turn off gradients for validation :
            with torch.no_grad():
                for images, labels in v_loader:
                    if torch.cuda.is_available():
                        images, labels = images.to('cuda') , labels.to('cuda')
                        model.to('cuda')
                        
                    output = model(images)
                    loss = criterion(output, labels)
                    tot_valid_loss += loss.item()

                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += equals.type(torch.FloatTensor).mean()

            # Get mean loss for train and validation sets
            train_loss = tot_train_loss / len(t_loader)
            valid_loss = tot_valid_loss / len(v_loader)

            print("Epoch: {}/{}  ".format(e+1, epochs),
                  "Training Loss: {:.3f}  ".format(train_loss),
                  "Validation Loss: {:.3f}  ".format(valid_loss),
                  "Validation Accuracy: {:.3f}".format(accuracy / len(v_loader)))

            model.train()
          
def train_data_transform(root):
    data_dir = root
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(data_dir + '/train', transform = train_transform)
    return train_data

def save_checkpoint(model,path='checkpoint_2.pth',structure ='vgg16', hidden_size = [4096], out_size = 102, dropout=0.5, lr=0.001, epochs=5):
    
    train_data = train_data_transform('flowers')    
    model.class_to_idx =  train_data.class_to_idx
    model.to ('cpu')
    
    # creating dictionary 
    checkpoint = {'structure' :structure,
                  'hidden_size':hidden_size,
                  'out_size':out_size,
                  'epochs' : epochs,
                  'dropout':dropout,
                  'lr':lr,
                  'state_dict': model.state_dict (),
                  'mapping_class_to_idx': model.class_to_idx,
                  'drop_out': 0.5}    
    # saving :
    torch.save(checkpoint, 'checkpoint_2.pth')