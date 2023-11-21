import torch
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        
        # Initialisation des dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Couche linéaire pour les entrées
        self.input_linear = nn.Linear(self.input_dim, self.hidden_dim)
        
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


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

