import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F 

from utils import RNN, device

BATCH_SIZE=32
## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))
ALPHABET_SIZE = len(id2lettre)

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]


with open('../../data/trump_full_speech.txt', 'r') as f:
    text = f.read()

ds_train = TrumpDataset(text)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)

# 2. Définition du modèle
input_size = ALPHABET_SIZE
latent = 64
model = RNN(input_size, latent, input_size).to(device)

# 3. Fonction de coût
criterion = torch.nn.CrossEntropyLoss()

# 4. Entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data, target in data_train:
        data, target = data.to(device), target.to(device)
        data = F.one_hot(data, num_classes=ALPHABET_SIZE).float()
        data = data.transpose(0,1)
        length, batch, _ = data.shape
        optimizer.zero_grad()
        h0 = torch.zeros(batch, latent).to(device)  # état caché initial
        h = model(data, h0)
        output = model.decode(h)
        output = output.transpose(0,1)

        target = F.one_hot(target, num_classes=ALPHABET_SIZE).float()
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.sum()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_train)}")

def flatten(l):
    return [item for sublist in l for item in sublist]

# 5. Génération de texte
def generate_text(model, start_string, length=100):
    model.eval()
    with torch.no_grad():
        input_seq = string2code(start_string).to(device)
        input_seq = F.one_hot(input_seq, num_classes=ALPHABET_SIZE).float()
        generated_text = start_string
        for _ in range(length):
            h = torch.zeros(1, latent).to(device)
            input_seq = input_seq.reshape(1, input_seq.size(0), input_seq.size(1))
            input_seq = input_seq.transpose(0,1)
            h = model(input_seq, h)
            output = model.decode(h[-1]) # dernière lettre
            probabilities = F.softmax(output[0], dim=0)
            predicted_id = torch.multinomial(probabilities, num_samples=1)
            if predicted_id.size(dim=0) == 1:
                predicted_id = predicted_id.item()
                generated_text += code2string([predicted_id])
            else:
                predicted_id = flatten(predicted_id.tolist())
                generated_text += code2string(predicted_id)
            
            input_seq = F.one_hot(string2code(generated_text).to(device), num_classes=ALPHABET_SIZE).float()
        return generated_text

print(generate_text(model, "The", 200))