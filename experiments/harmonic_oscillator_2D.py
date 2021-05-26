import torch
import numpy as np
from scipy.constants import hbar, m_e
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.helper import dfx, Sin
from models.model import EigenvalueProblemModel
from matplotlib.cm import coolwarm

# Solution on bounds
x_min = -6
x_max = 6

def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx[:,0])[:,0] + dfx(x, psi_dx[:,1])[:,1]
    diff = 1/2 * psi_ddx + (E-potential(x)) * psi # -hbar**2/(2*m_e) * psi_ddx + potential(x) * psi  - E * psi
    loss = torch.mean(diff**2)
    return loss

def potential(x):
    k = 4
    return k*(x[:, 0]**2 + x[:, 1]**2)/2

def compose_psi(x, N):
    f_b = 0
    dt = x - x_min
    psi = f_b + (1 - torch.exp(-dt[:, 0])) * (1 - torch.exp(-dt[:, 1])) * (1 - torch.exp(dt[:,0]-x_max*2)) * (1 - torch.exp(dt[:,1]-x_max*2)) * N.reshape(-1)
    return psi

def perturb(grid, x_min=-6, x_max=6, n_train=101, sig=0.05):
    noise = torch.randn_like(grid) * sig
    x = grid + noise
    # Make sure perturbation still lay in domain
    x[x[:,0] < x_min, 0] = x_min - x[x[:,0] < x_min, 0]
    x[x[:,0] > x_max, 0] = 2*x_max - x[x[:,0] > x_max, 0]
    x[x[:,1] < x_min, 1] = x_min - x[x[:,1] < x_min, 1]
    x[x[:,1] > x_max, 1] = 2*x_max - x[x[:,1] > x_max, 1]
    # Make sure at least one point is at the boundaries
    # x[:n_train,0] = torch.zeros_like(x[:n_train,1])
    # x[-n_train:,0] = torch.ones_like(x[:n_train,1])
    # x[0::n_train,1] = torch.zeros_like(x[::n_train,1])
    # x[n_train-1::n_train,1] = torch.ones_like(x[::n_train,1])

    return x

def driver(index):
    return 0.+0.25*(index//2000)

grid1D = torch.linspace(x_min, x_max, 20)
grid2D_x, grid2D_y = torch.meshgrid(grid1D,grid1D)
grid = torch.cat([grid2D_x.reshape(-1,1), grid2D_y.reshape(-1,1)], dim=1)

model = EigenvalueProblemModel([2, 20, 20, 20, 1], Sin, compose_psi, PDE_loss, lr=8e-3, start_eigenvalue=6.0)
model.train(driver, 2000, grid, perturb, int(2e4), max_required_loss=1e-2, rtol=0.01, fraction=6, reg_param=1e-3, pde_param=1)
model.plot_history()

n_plot = 100

grid1D = torch.linspace(x_min, x_max, n_plot)
grid2D_x, grid2D_y = torch.meshgrid(grid1D,grid1D)
grid = torch.cat([grid2D_x.reshape(-1,1), grid2D_y.reshape(-1,1)], dim=1)

marker = input("Plot eigenvalue, marker: ['q'] to quit   ")
while marker != 'q':
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    try:
        psi = model.get_eigenfunction(marker)
        Z = psi(grid).reshape(n_plot,n_plot)
        # Z = normalise2D(Z, dx, dy)
        surf = ax.plot_surface(grid2D_x.detach().numpy(),
                               grid2D_y.detach().numpy(),
                               Z.detach().numpy(),
                               cmap = coolwarm)
        plt.show()
    except TypeError:
        print("invalid input")
    finally:
        marker = input("Plot eigenvalue, marker: ['q'] to quit   ")