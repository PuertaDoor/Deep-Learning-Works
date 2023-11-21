import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    mask_output = (output != padcar) + 1e-10 # avoid negative values when calculating loss
    mask_target = (target != padcar) + 1e-10
    
    # Calculer la perte de cross-entropie
    loss = nn.CrossEntropyLoss(reduce="none")(output * mask_output, target * mask_target)
    
    return loss

# output = torch.randint(1, 5, (3,5,2))
# target = torch.randint(0, 2, (3,5))
# padcar = 1
# print(maskedCrossEntropy(output, target, padcar))


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        
        # Initialisation des dimensions
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.device = device

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Couche linéaire pour l'état d'entrée
        self.input_linear = nn.Linear(embedding_dim, self.hidden_dim)
        
        # Couche linéaire pour l'état caché
        self.hidden_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Couche de décodage
        self.decode_linear = nn.Linear(self.hidden_dim, self.output_dim)
        
    def one_step(self, x, h):
        """
        Traite un pas de temps.
        x : Entrée à l'instant t de taille batch × dim
        h : État caché à l'instant t de taille batch × latent
        Retourne : État caché suivant de taille batch × latent
        """
        return torch.tanh(self.input_linear(x) + self.hidden_linear(h))
    
    def forward(self, x, h):
        """
        Traite tout le batch de séquences.
        x : Entrée de taille length × batch × dim
        h : État caché initial de taille batch × latent
        Retourne : Séquence des états cachés de taille length × batch × latent
        """
        # Initialisation du tensor pour stocker la séquence des états cachés
        h_sequence = []

        # Traitement de chaque pas de temps
        for i in range(x.size(0)):
            h = self.one_step(x[i], h)
            h_sequence.append(h)

        return torch.stack(h_sequence)
    
    def decode(self, h):
        """
        Décode l'état caché.
        h : État caché de taille batch × latent
        Retourne : Tenseur décodé de taille batch × output
        """
        return self.decode_linear(h)


class GRU(RNN):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRU, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim)
        
        # Gates
        self.Wz = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.Wr = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.W = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        
    def one_step(self, x, h):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.Wz(hx))
        r = torch.sigmoid(self.Wr(hx))
        h_tilda = torch.tanh(self.W(torch.cat([r * h, x], dim=1)))
        h_next = (1 - z) * h + z * h_tilda
        return h_next

class LSTM(RNN):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__(vocab_size, embedding_dim, hidden_dim, output_dim)
        
        # Gates
        self.Wf = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.Wi = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.Wo = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.Wc = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        
    def one_step(self, x, h, c):
        hx = torch.cat([h, x], dim=1)
        f = torch.sigmoid(self.Wf(hx))
        i = torch.sigmoid(self.Wi(hx))
        o = torch.sigmoid(self.Wo(hx))
        c_tilda = torch.tanh(self.Wc(hx))
        c_next = f * c + i * c_tilda
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    
    def forward(self, x, h, c):
        """
        Traite tout le batch de séquences.
        x : Entrée de taille length × batch × dim
        h : État caché initial de taille batch × latent
        Retourne : Séquence des états cachés de taille length × batch × latent
        """
        # Initialisation du tensor pour stocker la séquence des états cachés
        h_sequence = []

        # Traitement de chaque pas de temps
        for i in range(x.size(0)):
            h, c = self.one_step(x[i], h, c)
            h_sequence.append(h)

        return torch.stack(h_sequence)


def train_model(model, data_loader, num_epochs=1, lr=1e-7):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            # On veut prédire le prochain caractère d'une chaîne
            data, target = batch[:-1].to(device), batch[1:].to(device)
            data = model.embedding(data)
            data = data.transpose(0,1)
            optimizer.zero_grad()
            if isinstance(model, LSTM):
                h0, c0 = torch.zeros(data.size(1), model.hidden_dim).to(device), torch.zeros(data.size(1), model.hidden_dim).to(device)
                h = model(data, h0, c0)
            else:
                h0 = torch.zeros(data.size(1), model.hidden_dim).to(device)
                h = model(data, h0)
            
            output = model.decode(h)
            output = output.transpose(0,1)
            
            target = model.embedding(target)

            loss = maskedCrossEntropy(output, target, PAD_IX)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    return losses

with open('../../data/trump_full_speech.txt', 'r') as f:
    text = f.read()

BATCH_SIZE = 32

ds_train = TextDataset(text)
data_train = DataLoader(ds_train, collate_fn=pad_collate_fn, batch_size=BATCH_SIZE, shuffle=True)

# 2. Définition du modèle
vocab_size = len(LETTRES) + 2  # +1 pour le caractère NULL

# rnn_model = RNN(vocab_size, embedding_dim=vocab_size, hidden_dim=50, output_dim=vocab_size).to(device)
# rnn_losses = train_model(rnn_model, data_train)

gru_model = GRU(vocab_size, embedding_dim=vocab_size, hidden_dim=vocab_size, output_dim=vocab_size).to(device)
gru_losses = train_model(gru_model, data_train)

# lstm_model = LSTM(vocab_size, embedding_dim=vocab_size, hidden_dim=50, output_dim=vocab_size).to(device)
# lstm_losses = train_model(lstm_model, data_train)

# print("RNN Generated Text:", generate(rnn_model, rnn_model.embedding, rnn_model.decode, EOS_IX, "The", 200))
# print("GRU Generated Text:", generate(gru_model, gru_model.embedding, gru_model.decode, EOS_IX, "The", 200))
# print("LSTM Generated Text:", generate(lstm_model, lstm_model.embedding, lstm_model.decode, EOS_IX, "The", 200, lstm=True))

# print("RNN Generated Text:", generate_beam(rnn_model, EOS_IX, 10, "The", 200))
# print("GRU Generated Text:", generate_beam(gru_model, gru_model.embedding, gru_model.decode, EOS_IX, 10, "The", 200))
# print("LSTM Generated Text:", generate_beam(lstm_model, EOS_IX, 10, "The", 200))

# writer = SummaryWriter()

# for name, param in rnn_model.named_parameters():
#     writer.add_histogram(name, param.grad, 0)
#     writer.add_histogram(f"{name}_data", param, 0)

# """
# for name, param in gru_model.named_parameters():
#     writer.add_histogram(name, param.grad, 0)
#     writer.add_histogram(f"{name}_data", param, 0)

# for name, param in lstm_model.named_parameters():
#     writer.add_histogram(name, param.grad, 0)
#     writer.add_histogram(f"{name}_data", param, 0)
# """

# writer.close()