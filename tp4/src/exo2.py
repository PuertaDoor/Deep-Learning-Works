from utils import RNN, device,SampleMetroDataset
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 2
#Taille du batch
BATCH_SIZE = 32

PATH = "../../data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test=SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


# 1. Définir le modèle RNN
latent = 64
model = RNN(DIM_INPUT, latent, CLASSES).to(device)

# 2. Définir le critère de perte
criterion = torch.nn.CrossEntropyLoss()

# 3. Définir l'optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entraîner le modèle
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, target in data_train:
        data, target = data.to(device), target.to(device)
        data = data.transpose(0,1) # FROM batch x length x dim TO length x batch x dim
        length, batch, dim = data.shape
        optimizer.zero_grad()
        h0 = torch.zeros(batch, latent).to(device)  # état caché initial
        output = model(data, h0)
        output = model.decode(output[-1]) # On décode uniquement hT (many-to-one)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_train)}")

# 5. Évaluer le modèle
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in data_test:
        data, target = data.to(device), target.to(device)
        data = data.transpose(0,1) # FROM batch x length x dim TO length x batch x dim
        length, batch, dim = data.shape
        h0 = torch.zeros(batch, latent).to(device)  # état caché initial
        output = model(data, h0)
        output = model.decode(output[-1])
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy on test data: {100 * correct / total}%")