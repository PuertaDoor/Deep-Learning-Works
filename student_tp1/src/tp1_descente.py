import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context

mse = MSE()
lin = Linear()


# Les données supervisées
x = torch.randn(10, 7)
y = torch.randn(10, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(7, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()

for n_iter in range(100):
    ctx_mse = Context()
    ctx_lin = Context()

    ##  TODO:  Calcul du forward (loss)
    loss = MSE.forward(ctx_mse, lin.forward(ctx_lin, x, w, b), y)

    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    grad_yhat, _ = MSE.backward(ctx_mse, 1)
    _, grad_w, grad_b = Linear.backward(ctx_lin, grad_yhat)

    ##  TODO:  Mise à jour des paramètres du modèle
    w = w - epsilon * grad_w
    b = b - epsilon * grad_b
