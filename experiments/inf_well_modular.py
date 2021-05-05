from models.model import EigenvalueProblemModel
from models.helper import perturb1D, Sin, dfx
import torch
from functools import partial
L = 1
x_min = 0
x_max = L

def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx)
    diff = 1/2 * psi_ddx + E * psi # -hbar**2/(2*m_e) * psi_ddx + potential(x) * psi  - E * psi
    loss = torch.mean(diff**2)
    return loss

def compose_psi(x, N):
    f_b = 0
    dt = x - x_min
    psi = f_b + (1 - torch.exp(-dt)) * (1 - torch.exp(dt-x_max)) * N
    return psi

def driver(index):
    return -4 + (index // 2500)

grid = torch.linspace(x_min, x_max, 100).reshape(-1, 1)

model = EigenvalueProblemModel([1, 10, 10, 1], Sin, compose_psi, PDE_loss)
model.train(driver, grid, partial(perturb1D, x_min=x_min, x_max=x_max), int(125e3), 100, 1)
model.plot_history()