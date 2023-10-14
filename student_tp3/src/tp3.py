from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
# Téléchargement des données

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist");
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)

class MonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data).view(-1, 784)
        self.labels = torch.tensor(labels).view(-1, 784)

    def __getitem__(self, index):
        
        flattened_data = self.data[index]
        normalized = (flattened_data - flattened_data.min().item()) / (flattened_data.max().item() - flattened_data.min().item())
        if self.data.shape == self.labels.shape:
            flattened_labels = self.labels[index]
            self.labels[index] = (flattened_labels - flattened_labels.min().item()) / (flattened_labels.max().item() - flattened_labels.min().item())
        
        return (normalized, self.labels[index])

    def __len__(self):
        return self.data.size(0)

# BATCH_SIZE = 1
# data = DataLoader(MonDataset(train_images, train_images), shuffle=True, batch_size=1)
# for x,y in data:
#    print(x,y)


class AutoEncoder(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input, output),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(output, input),
            nn.Sigmoid()
        )
    
    def forward(self, datax):
        enc = self.encoder(datax)
        writer.add_embedding(enc)
        dec = self.decoder(enc)

        return dec
    

savepath = Path("model.pch")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch, self.iteration = 0,0

if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)

else:
    data = DataLoader(MonDataset(train_images, train_images), shuffle=True, batch_size=2)
    model = AutoEncoder(784, 20)
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    state = State(model, optim)

for epoch in range(state.epoch, 10):
    for x,y in data:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x)
        l = nn.MSELoss()(xhat, x) # even though it is not the most optimal one
        l.backward()
        state.optim.step()
        writer.add_scalar('Loss', l.item(), state.iteration)
        writer.add_image('Decoded Images', make_grid(xhat), state.iteration)
        state.iteration += 1
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)


# model = AutoEncoder(784,128)
# z1, z2 = model.encoder(torch.tensor(test_images[0]).view(-1).float().to(device)), model.encoder(torch.tensor(test_images[1]).view(-1).float().to(device))
# lambdas = torch.linspace(0, 1, steps=10)  # 10 valeurs entre 0 et 1
# interpolated_images = []
# for lambda_ in lambdas:
#    z_interpolated = lambda_ * z1 + (1 - lambda_) * z2
#    x_interpolated = model.decoder(z_interpolated)
#    interpolated_images.append(x_interpolated)
    
# 'interpolated_images' est maintenant une liste d'images interpolées
# interpolated_images_tensor = torch.stack(interpolated_images).unsqueeze(1).repeat(1,3,1,1).double()/255.
# writer.add_image('Interpolated Images', make_grid(interpolated_images_tensor), state.iteration)

class HighwayLayer(nn.Module):
    def __init__(self, size, gate_activation=F.sigmoid, transform_activation=F.relu):
        super(HighwayLayer, self).__init__()
        self.size = size
        self.gate_activation = gate_activation
        self.transform_activation = transform_activation
        
        self.transform_weights = nn.Linear(size, size)
        self.gate_weights = nn.Linear(size, size)
        
    def forward(self, x):
        transform = self.transform_activation(self.transform_weights(x))
        gate = self.gate_activation(self.gate_weights(x))
        highway = gate * transform + (1 - gate) * x
        return highway

class HighwayNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(HighwayNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.initial_layer = nn.Linear(input_size, output_size)
        self.highway_layers = nn.ModuleList([HighwayLayer(output_size) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.initial_layer(x)
        for layer in self.highway_layers:
            x = layer(x)
        return x
    

savepath = Path("highway.pch")

if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)

else:
    data = DataLoader(MonDataset(train_images, train_images), shuffle=True, batch_size=2)
    model = HighwayNetwork(784, 784, 5)
    model = model.to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    state = State(model, optim)

for epoch in range(state.epoch, 10):
    for x,y in data:
        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x)
        l = nn.MSELoss()(xhat, x) # even though it is not the most optimal one
        l.backward()
        state.optim.step()
        writer.add_scalar('Loss', l.item(), state.iteration)
        writer.add_image('Decoded Images', make_grid(xhat), state.iteration)
        state.iteration += 1
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state,fp)