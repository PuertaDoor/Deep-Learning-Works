import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re

## Token de padding (BLANK)
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i]).tolist()

def pad_collate_fn(samples: List[List[int]]):
    """
    Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.
    """
    # Ajouter le code EOS à la fin de chaque exemple
    samples = [sample + [EOS_IX] for sample in samples]
    
    # Trouver la longueur maximale parmi les échantillons
    max_len = max(len(sample) for sample in samples)
    
    # Padder chaque échantillon avec le code du caractère nul pour qu'ils aient tous la même longueur
    padded_samples = [sample + [PAD_IX] * (max_len - len(sample)) for sample in samples]
    
    # Convertir en tenseur
    batch = torch.tensor(padded_samples)

    # Transposer le batch pour avoir longueur x taille du batch
    batch = batch.t()
    
    return batch

if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    print("Chaîne à code : ", test)
    print(data.shape)
    # Longueur maximum
    assert data.shape == (7, 3)
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5,2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:,1]==0).sum() == data.shape[0]-4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
