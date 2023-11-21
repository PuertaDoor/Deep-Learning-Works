import itertools
import logging
from tqdm import tqdm

from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)

ds = prepare_dataset('org.universaldependencies.french.gsd')


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
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

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))


logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)


logging.info("Vocabulary size: %d", len(words))


BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)




class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Embedding layer that turns word indexes into embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs and outputs hidden states with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer maps from the hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # Compute the lengths of the sequences
        lengths = torch.sum(sentence != Vocabulary.PAD, dim=1)

        # Convert word indexes to embeddings
        embeds = self.word_embeddings(sentence)

        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)

        # Run the LSTM on the packed embeddings
        lstm_out, _ = self.lstm(packed_embeds)

        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Map the output of the LSTM to tag space
        tag_space = self.hidden2tag(lstm_out)

        # Convert to log probabilities for loss calculation
        tag_scores = F.log_softmax(tag_space, dim=1)

        return tag_scores
    
def predict_tags(sentence: List[str], words_vocab: Vocabulary, tags_vocab: Vocabulary, model: nn.Module):
    # Ensure the model is in evaluation mode
    model.eval()

    # Convert the sentence to a tensor of word indices
    sentence_indices = [words_vocab.get(word, adding=False) for word in sentence]
    sentence_tensor = torch.LongTensor(sentence_indices).unsqueeze(0)  # Add batch dimension

    # Pass the sentence through the model
    with torch.no_grad():
        tag_scores = model(sentence_tensor)

    # Convert the output scores to tag indices
    _, predicted_tag_indices = torch.max(tag_scores, dim=2)

    # Convert the tag indices to tag names
    predicted_tags = [tags_vocab.getword(idx.item()) for idx in predicted_tag_indices.flatten()]

    return predicted_tags

# Instantiate the model, the loss function, and the optimizer
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

model = LSTMTagger(len(words), len(tags), EMBEDDING_DIM, HIDDEN_DIM)
loss_function = nn.CrossEntropyLoss(ignore_index=Vocabulary.PAD)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10): # number of epochs is a hyperparameter you can tune
    total_loss = 0
    for sentence, tag in train_loader:

        # Reset gradients
        model.zero_grad()

        # Prepare the inputs to be passed to the model (i.e., turn the words into integer indices and wrap them in tensors)
        sentence_in = torch.tensor(sentence, dtype=torch.long)
        targets = torch.tensor(tag, dtype=torch.long)

        # Run forward pass.
        tag_score = model(sentence_in)

        # Flatten the tag_scores and tags for computing the loss
        tag_score = tag_score.view(-1, tag_score.shape[-1])  # [batch_size * seq_len, num_tags]
        targets = targets.view(-1)  # [batch_size * seq_len]

        # Compute the loss, gradients, and update the parameters by calling optimizer.step()
        loss = loss_function(tag_score, targets)
        loss.backward()
        optimizer.step()

        # Keep track of the total loss for this epoch
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader)}")

# Example usage:
sentence = ["Ceci", "est", "une", "phrase", "de", "test"]
predicted_tags = predict_tags(sentence, words, tags, model)
print(predicted_tags)