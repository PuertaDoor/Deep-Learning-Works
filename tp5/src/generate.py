from textloader import  code2string, string2code, id2lettre
import math
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten(l):
    return [item for sublist in l for item in sublist]

def generate(rnn, emb, decoder, eos, start="", maxlen=200, lstm=False):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    rnn.eval()
    with torch.no_grad():
        raw_seq = string2code(start).to(device)
        generated_text = start
        for _ in range(maxlen):
            h0 = torch.zeros(1, rnn.hidden_dim).to(device)
            input_seq = emb(raw_seq)
            input_seq = input_seq.reshape(1, input_seq.size(0), input_seq.size(1))
            input_seq = input_seq.transpose(0,1)
            if lstm:
                c0 = torch.zeros(1, rnn.hidden_dim).to(device)
                h = rnn(input_seq, h0, c0)
            else:
                h = rnn(input_seq, h0)
            output = decoder(h[-1]) # dernière lettre
            probabilities = F.softmax(output[0], dim=0)
            predicted_id = torch.multinomial(probabilities, num_samples=1)
            if predicted_id.size(dim=0) == 1:
                predicted_id = predicted_id.item()
                if predicted_id == eos:
                    break
                generated_text += code2string([predicted_id])
            else:
                predicted_id = flatten(predicted_id.tolist())

                if eos in predicted_id:
                    break

                generated_text += code2string(predicted_id)
            
            raw_seq = string2code(generated_text).to(device)

        return generated_text

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200, lstm=False):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    rnn.eval()
    with torch.no_grad():
        raw_seq = string2code(start).to(device)
        generated_text = start

        # Initialize beam search
        sequences = [([raw_seq], 0)]  # (sequence, log probability)
        end_candidates = []

        for _ in range(maxlen):
            all_candidates = []

            for sequence, score in sequences:
                h0 = torch.zeros(1, rnn.hidden_dim).to(device)
                input_seq = emb(torch.tensor(sequence[-1]).clone().detach().to(device))
                input_seq = input_seq.unsqueeze(0)
                input_seq = input_seq.transpose(0, 1)
                print(input_seq.shape)

                if lstm:
                    c0 = torch.zeros(1, rnn.hidden_dim).to(device)
                    h = rnn(input_seq, h0, c0)
                else:
                    print(h0.shape, input_seq.shape)
                    h = rnn(input_seq, h0)

                output = decoder(h[-1])
                probabilities = F.softmax(output[0], dim=0)
                top_probabilities, top_indices = torch.topk(probabilities, k)

                for i in range(k):
                    predicted_id = top_indices[i].item()
                    new_sequence = sequence + [predicted_id]
                    new_score = score + torch.log(top_probabilities[i]).item()

                    if predicted_id == eos:
                        end_candidates.append((new_sequence, new_score))
                    else:
                        all_candidates.append((new_sequence, new_score))

            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = all_candidates[:k]

        # If no sequence reaches eos, consider end_candidates
        if not sequences:
            sequences = end_candidates

        # Choose the sequence with the highest score
        best_sequence = max(sequences, key=lambda x: x[1])[0]
        generated_text = "".join(code2string(token) for token in best_sequence)

        return generated_text


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        logits = decoder(h)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs < alpha
        selected_indices = sorted_indices[mask]
        
        new_probs = torch.zeros_like(probs)
        new_probs[selected_indices] = probs[selected_indices]
        new_probs /= new_probs.sum()  # Renormaliser
        
        return new_probs
    return compute
