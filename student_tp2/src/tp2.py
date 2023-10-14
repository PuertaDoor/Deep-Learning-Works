import os
import torch
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from torch.nn import Tanh, MSELoss, Sequential, Linear




# -------------- QUESTION 1 --------------

mse = MSELoss()

if not os.path.exists('runs/'):
    os.mkdir('runs')


data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

epsilon = 1e-6

def gradient_descent(datax, datay, n_iters, epsilon):
    # Les dimensions d'entrée et de sortie du module linéaire
    input_size = datax.shape[1]
    output_size = datay.shape[1]

    lin = Linear(input_size, output_size)

    writer = SummaryWriter('runs/GD')

    for n_iter in range(n_iters):

        # Calcul du forward (loss)
        yhat = lin(datax)
        loss = mse.forward(yhat, datay)

        # on peut visualiser avec
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train', loss, n_iter)

        # Sortie directe
        print(f"Itérations {n_iter}: loss {loss}")

        # Calcul du backward (grad_w, grad_b)
        loss.backward()
        
        with torch.no_grad():
            for param in lin.parameters():
                param -= epsilon * param.grad
            lin.zero_grad()

gradient_descent(datax, datay, 100, epsilon)

def stochastic_gradient_descent(datax, datay, n_iters, epsilon):
    # Les dimensions d'entrée et de sortie du module linéaire
    input_size = datax.shape[1]
    output_size = datay.shape[1]

    lin = Linear(input_size, output_size)
    writer = SummaryWriter('runs/SGD')

    for n_iter in range(n_iters):
        # Choisissez un exemple aléatoire
        idx = torch.randint(0, datax.size(0), (1,))
        x = datax[idx]
        y = datay[idx]

        yhat = lin(x)
        loss = mse.forward(yhat, y)
        writer.add_scalar('Loss/train', loss.item(), n_iter)

        print(f"Itérations {n_iter}: loss {loss.item()}")

        loss.backward()
        with torch.no_grad():
            for param in lin.parameters():
                param -= epsilon * param.grad
            lin.zero_grad()

stochastic_gradient_descent(datax, datay, 100, epsilon)

def minibatch_gradient_descent(datax, datay, n_iters, epsilon, batch_size):
    # Les dimensions d'entrée et de sortie du module linéaire
    input_size = datax.shape[1]
    output_size = datay.shape[1]

    lin = Linear(input_size, output_size)
    writer = SummaryWriter('runs/MBGD')

    for n_iter in range(0, n_iters, batch_size):
        # Extraction du minibatch
        batch_x = datax[n_iter:n_iter+batch_size]
        batch_y = datay[n_iter:n_iter+batch_size]

        yhat = lin(batch_x)
        loss = mse.forward(yhat, batch_y)
        writer.add_scalar('Loss/train', loss.item(), n_iter // batch_size)

        print(f"Itérations {n_iter // batch_size}: loss {loss.item()}")

        loss.backward()
        with torch.no_grad():
            for param in lin.parameters():
                param -= epsilon * param.grad
            lin.zero_grad()

minibatch_gradient_descent(datax, datay, 100, epsilon, 20)

# Vitesse de convergence : minibatch beaucoup plus rapide (5 itérations au lieu de 100)
# Résultats : moins concluants chez SGD (d'ailleurs la courbe était beaucoup plus bruitée), du pareil au même chez MBGD et GD.





# --------------- QUESTION 2 ---------------

EPS = 1e-3

w = torch.nn.Parameter(torch.randn(datax.shape[1], datay.shape[1]))
b = torch.nn.Parameter(torch.randn(w.shape[1]))

my_nn = Sequential(
    torch.nn.Linear(datax.shape[1], 2*datax.shape[1]),
    Tanh(),
    torch.nn.Linear(2*datax.shape[1], datay.shape[1])
)

optim = torch.optim.SGD(params=my_nn.parameters(),lr=EPS) ## on optimise selon w et b, lr : pas de gradient
optim.zero_grad() # Reinitialisation du gradient

NB_EPOCH = 100

criterion = MSELoss()

writer = SummaryWriter('runs/Sequential')

# Reinitialisation du gradient
for i in range(NB_EPOCH):
    y_pred = my_nn(datax)
    loss = criterion(y_pred, datay) #Calcul du cout

    loss.backward() # Retropropagation
    if i % 1 == 0:
        writer.add_scalar('Loss/train', loss.item(), i)
        print(f"Itérations {i}: loss {loss.item()}")
        optim.step() # Mise-à-jour des paramètres w et b
        optim.zero_grad() # Reinitialisation du gradient