import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from scipy.constants import hbar, m_e, epsilon_0, m_p, elementary_charge
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.helper import dfx, Sin
from models.model import EigenvalueProblemModel
from matplotlib.cm import coolwarm


radius_hydrogen = 31e-12
bound = 3*radius_hydrogen


mu = m_e * m_p / (m_e + m_p)
prefactor_laplacian = -hbar**2/(2*mu)
prefactor_potential = -elementary_charge**2/(4 * np.pi * epsilon_0)

def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx[:,0])[:,0] + dfx(x, psi_dx[:,1])[:,1] + dfx(x, psi_dx[:, 2])[:, 2]
    diff = (prefactor_laplacian * psi_ddx + (potential(x) - E)*psi)
    loss = torch.mean(diff**2)
    return loss

def potential(x):
    return prefactor_potential * 1/torch.sqrt(torch.sum(x**2, dim=1))


def compose_psi(x, N):
    #dt = x + bound
    #psi = (1 - torch.exp(-dt[:, 0])) * (1 - torch.exp(-dt[:, 1])) * (1 - torch.exp(-dt[:, 2])) * (1 - torch.exp(dt[:,0]-bound*2)) * (1 - torch.exp(dt[:,1]-bound*2)) * (1 - torch.exp(dt[:,2]-bound*2)) * N.reshape(-1)
    return N.reshape(-1)


def perturb(grid):
    sig = radius_hydrogen/5
    x_min = -bound
    x_max = bound
    noise = torch.randn_like(grid) * sig
    x = grid + noise
    # Make sure perturbation still lay in domain
    x[x[:,0] < x_min, 0] = x_min - x[x[:,0] < x_min, 0]
    x[x[:,0] > x_max, 0] = 2*x_max - x[x[:,0] > x_max, 0]
    x[x[:,1] < x_min, 1] = x_min - x[x[:,1] < x_min, 1]
    x[x[:,1] > x_max, 1] = 2*x_max - x[x[:,1] > x_max, 1]
    x[x[:,2] < x_min, 2] = x_min - x[x[:,2] < x_min, 2]
    x[x[:,2] > x_max, 2] = 2*x_max - x[x[:,2] > x_max, 2]
    return x

def driver(index):
    return -3.5e-18+0.1e-18*(index//1000)

grid1D = torch.linspace(-bound, bound, 5)
grid_x, grid_y, grid_z = torch.meshgrid(grid1D,grid1D, grid1D)
grid = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1, 1)], dim=1)



model = EigenvalueProblemModel([3, 20, 20, 20, 20, 1], Sin, compose_psi, PDE_loss, lr=1e-20, start_eigenvalue=-2e-18)
model.train(driver, 2000, grid, perturb, int(8e3), max_required_loss=1e-2, rtol=0.01, fraction=6, reg_param=1e-3, pde_param=1)
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