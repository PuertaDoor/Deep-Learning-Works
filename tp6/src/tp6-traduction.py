import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string

from pathlib import Path
from typing import List
import random
import time
import re
from torch.utils.tensorboard import SummaryWriter




logging.basicConfig(level=logging.INFO)

FILE = "../data/en-fra.txt"



def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=32

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)





class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input) 
        return self.gru(embedded)


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output)
        # output = self.softmax(output)
        return output, hidden

    def generate(self, enc_output, hidden, lenseq = 100, target = None):
        batch_size = enc_output.size(1)
        input_seq = torch.empty(1, batch_size, dtype=torch.long, device=device).fill_(Vocabulary.SOS) # SOS token

        output_seq = []

        for i in range(lenseq):
            output, hidden = self.forward(input_seq, hidden)
            
            if(target != None): #Teacher forcing
                input_seq = target[i,:].unsqueeze(0)
            else: #No teacher forcing
                topv, topi = output.topk(1)
                next_token = topi.squeeze().detach()
            
            output_seq.append(output)
        return torch.cat(output_seq, dim=0),hidden




def train_test_iter(loader,
          encoder, 
          decoder, 
          encoder_optimizer, 
          decoder_optimizer, 
          loss_fn, 
          train = True,
          max_length=MAX_LEN):
    
    t_loss = 0
    if(train):
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval() 
    
    #Initialisation hidden
    hidden = torch.zeros(1, BATCH_SIZE, HIDDEN_SIZE).to(device)
    for x, _, y, _ in loader:
        x, y = x.to(device), y.to(device)
        #Encodage
        y_enc, hidden = encoder(x, hidden)

        #Decodage avec ou sans Teacher Forcing
        tf = random.randint(0, 1)
        if(tf):
          y_dec, hidden = decoder.generate(y_enc, hidden, lenseq = y.size(0), target = y)
        else:
          y_dec, hidden = decoder.generate(y_enc, hidden, lenseq = y.size(0))
        y_dec = y_dec.transpose(1, 2)
      
        loss = loss_fn(y_dec, y)
        t_loss += loss.item()
        
        if(train):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
      
    #Exemple de traduction sur les dernières donnnées traitées
    prediction = vocFra.getwords(y_dec.argmax(1)[:, 0])
    print(f"\n     Source : {vocEng.getwords(x[:, 0])}\n",
        f"Prédiction : {prediction}\n",
        f"Traduction : {vocFra.getwords(y[:,0])}\n",
        f"Loss : {t_loss / len(loader)}\n\n")






LR = 0.01
EPOCHS = 10
HIDDEN_SIZE = 64
EMBED_SIZE = 64
FR_SIZE = vocFra.__len__()
ENG_SIZE = vocEng.__len__()

encoder = EncoderRNN(input_size=ENG_SIZE, embedding_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE).to(device)
decoder = DecoderRNN(hidden_size=HIDDEN_SIZE, embedding_size=EMBED_SIZE, output_size=FR_SIZE).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)

loss_fn = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)

# Iterate through epochs
for epoch in range(EPOCHS):  # number of epochs is a hyperparameter you can tune
    #Training
    print(f"EPOCH {epoch} -- TRAINING\n")
    train_test_iter(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn)
    #Test
    if((epoch+1)%5 == 0):
      print("TEST \n")
      train_test_iter(test_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn, train = False)
