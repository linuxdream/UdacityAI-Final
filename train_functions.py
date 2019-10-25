# Import libraries
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json

"""
load_data function takes a data directory and processes any transforms and data loading.
"""
def load_data(data_dir):
    # Set train and test dirs.
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Element zero will be for training while element one will be for the others.
    data_transforms = [transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(223),
                                    transforms.RandomRotation(180),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                       transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(223),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
                      ]

    # Load the data
    image_datasets = [
                        datasets.ImageFolder(train_dir, transform=data_transforms[0]),
                        datasets.ImageFolder(valid_dir, transform=data_transforms[1]),
                        datasets.ImageFolder(test_dir, transform=data_transforms[1])
                     ]

    # Define the data loaders
    dataloaders = [
                    torch.utils.data.DataLoader(image_datasets[0], batch_size=32, shuffle=True),
                    torch.utils.data.DataLoader(image_datasets[1], batch_size=32, shuffle=True),
                    torch.utils.data.DataLoader(image_datasets[2], batch_size=32, shuffle=True)
                  ]
    
    return dataloaders, image_datasets

"""
load_model function loads and sets up a model for training.
"""
def load_model(arch="densenet121", gpu=True, learning_rate=0.001, hidden_units=[500, 200], input_size=1024, output_size=102, dropout=0.20):
    if arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    else:
        print("Unsupported model specified. densenet121, vgg16, and alexnet are supported.")
        exit(1)
    
    # Don't train the other layers
    for param in model.parameters():
        param.requires_grad = False

    # Build the classifier
    hidden_layers = {}
    
    # Loop through the hidden layers and keep building
    for idx, layer in enumerate(hidden_units):
        if(idx == 0):
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(input_size, layer)
        else:
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(hidden_units[idx-1], layer)
            
        hidden_layers[f'relu{idx+1}'] = nn.ReLU()
        hidden_layers[f'dropout {idx+1}'] = nn.Dropout(p=dropout)

    # Final layer (output)
    final_layer_count = len(hidden_units) + 1
    hidden_layers[f'Layer {final_layer_count}'] = nn.Linear(hidden_units[-1], output_size)
    hidden_layers[f'output'] = nn.LogSoftmax(dim=1)
    
    # Assign the new classifier
    model.classifier = nn.Sequential(OrderedDict(hidden_layers))

    # Specify the criterion
    criterion = nn.NLLLoss();

    # Specify our optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Track which device is available
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Apply that device
    model.to(device)
    
    return model, optimizer, criterion

"""
train_model actually trains the model
"""
def train_model(model, criterion, optimizer, epochs, dataloaders, gpu):
    # Specify params
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 10
    
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")

    # Loop through each epoch
    for epoch in range(epochs):
        for inputs, labels in dataloaders[0]:
            steps += 1
            
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders[2]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders[2]):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders[2]):.3f}")
                running_loss = 0
                model.train()
                
    return model, optimizer, criterion
    
"""
Save the model checkpoint
"""
def save_model(save_dir, model, optimizer, criterion, arch, input_size, output_size, hidden_layers, epochs, image_idx):
    checkpoint = {
        'arch': arch,
        'input_size': int(input_size),
        'output_size': int(output_size),
        'hidden_layers': hidden_layers,
        'epochs': int(epochs),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict(),
        'class_to_idx': image_idx,
        'loss': criterion
    }

    torch.save(checkpoint, save_dir + '/cli_checkpoint.pth')
    
    
def start(args):
    # Load the data first
    dataloaders, image_datasets = load_data(args.data_dir)

    # Setup the network
    model, optimizer, criterion = load_model(arch=args.arch, gpu=args.gpu, learning_rate=args.learning_rate, hidden_units=args.hidden_units, output_size=args.output_size, input_size=args.input_size)

    # Train the network
    model, optimizer, criterion = train_model(model=model, criterion=criterion, optimizer=optimizer, epochs=args.epochs, dataloaders=dataloaders, gpu=args.gpu)
    
    # Save model
    save_model(save_dir=args.save_dir, model=model, optimizer=optimizer, criterion=criterion, arch=args.arch, input_size=args.input_size, output_size=args.output_size, hidden_layers=args.hidden_units, epochs=args.epochs, image_idx=image_datasets[0].class_to_idx)