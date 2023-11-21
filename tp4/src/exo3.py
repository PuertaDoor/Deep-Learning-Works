from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "../../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# Initialisation du modèle, critère de perte et optimiseur
latent = 64
model = RNN(DIM_INPUT, latent, DIM_INPUT).to(device)
criterion = torch.nn.MSELoss() # Coût plus adapté
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entraînement
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, target in data_train:
        data, target = data.to(device), target.to(device)
        data = data.transpose(0,1) # FROM batch x length x dim TO length x batch x dim
        length, batch, stations, dim = data.shape
        yhat = torch.zeros(batch, length, stations, dim).to(device) # Yhat init for loss
        optimizer.zero_grad()
        for station in range(stations):
            h0 = torch.zeros(batch, latent).to(device)
            tmp_data = data[:,:,station,:]
            output = model(tmp_data, h0)
            yhat_station = []
            for i in range(output.size(1)): # many-to-many
                yhat_station.append(model.decode(output[:,i]))
            yhat[:,:,station,:] = torch.stack(yhat_station)

        loss = criterion(yhat, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_train)}")

# Évaluation
model.eval()
total_loss = 0
with torch.no_grad():
    for data, target in data_test:
        data, target = data.to(device), target.to(device)
        data = data.transpose(0,1) # FROM batch x length x dim TO length x batch x dim
        length, batch, stations, dim = data.shape
        yhat = torch.zeros(batch, length, stations, dim).to(device) # Yhat init for loss
        for station in range(stations):
            h0 = torch.zeros(batch, latent).to(device)
            tmp_data = data[:,:,station,:]
            output = model(tmp_data, h0)
            yhat_station = []
            for i in range(output.size(1)): # many-to-many avec t+length prédiction
                yhat_station.append(model.decode(output[:,i]))
            yhat[:,:,station,:] = torch.stack(yhat_station)

        loss = criterion(yhat, target)
        total_loss += loss.item()

print(f"Loss on test data: {total_loss/len(data_test)}")
