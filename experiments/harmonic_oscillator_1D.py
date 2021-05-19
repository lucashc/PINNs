import torch
import numpy as np
from scipy.constants import hbar, m_e
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.helper import dfx, Sin, perturb1D
from models.model import EigenvalueProblemModel
from functools import partial


# Solution on bounds
x_min = -6
x_max = 6

def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx)
    diff = 1/2 * psi_ddx + (E-potential(x)) * psi # -hbar**2/(2*m_e) * psi_ddx + potential(x) * psi  - E * psi
    loss = torch.mean(diff**2)
    return loss

def potential(x):
    k = 4
    return k*x**2/2

def compose_psi(x, N):
    f_b = 0
    dt = x - x_min
    psi = f_b + (1 - torch.exp(-dt)) * (1 - torch.exp(dt-x_max*2)) * N
    return psi

    return x

def driver(index):
    return -2+0.25*(index//2000)

grid = torch.linspace(x_min, x_max, 400).reshape(-1, 1)

model = EigenvalueProblemModel([1, 50, 50, 1], Sin, compose_psi, PDE_loss, lr=8e-3, start_eigenvalue=1.0)
model.train(driver, 1000, grid, partial(perturb1D, x_min=x_min, x_max=x_max), int(4e5), max_required_loss=1e-2, rtol=0.001, fraction=3)
model.plot_history()

large_grid = torch.linspace(x_min, x_max, 400).reshape(-1, 1)

for marker in model.eigenfunctions.keys():
    y = model.get_eigenfunction(marker)(large_grid)
    plt.plot(large_grid.reshape(-1).numpy(), y.reshape(-1).detach().numpy())
    plt.xlabel("x")
    plt.ylabel("$\Psi(x)$")
    plt.xlim(x_min, x_max)
    plt.grid()
    plt.title(f"Eigenfunction {marker} with energy {model.eigenfunctions[marker][1]}")
    plt.show() 