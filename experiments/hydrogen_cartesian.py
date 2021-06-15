import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from scipy.constants import hbar, m_e, m_p, elementary_charge, value
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.helper import dfx, Sin
from models.model import EigenvalueProblemModel
from matplotlib.cm import coolwarm
import dill

# Constants

# Prefactors
bohr_radius = value("Bohr radius")
laplacian_prefactor = -0.5 * (m_e + m_p) / m_p
energy_prefactor = m_e * bohr_radius**2 / hbar**2 * elementary_charge

# Bounds


# Characteristic energy
E_0 = -13.6 # eV


def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx[:,0])[:,0] + dfx(x, psi_dx[:,1])[:,1] + dfx(x, psi_dx[:, 2])[:, 2]
    r = torch.sqrt(torch.sum(x**2, dim=1))
    diff = laplacian_prefactor*psi_ddx - 1/r * psi - energy_prefactor * E * psi
    loss = torch.mean(diff**2)
    return loss

def compose_psi(x, N):
    radius = torch.sqrt(torch.sum(x**2, dim=1))
    return (1-torch.exp(radius - bohr_radius*2)) * N.reshape(-1)


def perturb(grid):
    noise = torch.randn_like(grid) * bohr_radius/8
    x = grid + noise
    # Make sure perturbation still lay in domain
    r_2 = torch.sum(x**2, dim=1)
    idx = r_2 > (bohr_radius)**2
    x[idx,0] = bohr_radius**2 / r_2[idx] * x[idx,0]
    x[idx,1] = bohr_radius**2 / r_2[idx] * x[idx,1]
    x[idx,2] = bohr_radius**2 / r_2[idx] * x[idx,2]

    return x

def driver(index):
    return 2*E_0-0.01*E_0*(index//1000)

grid1D = torch.linspace(0, bohr_radius, 10)
grid_x, grid_y, grid_z = torch.meshgrid(grid1D,grid1D, grid1D)
grid = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1, 1)], dim=1)

model = EigenvalueProblemModel([3, 20, 20, 20, 1], Sin, compose_psi, PDE_loss, lr=8e-3, start_eigenvalue=1.2*E_0)
model.train(driver, 2000, grid, perturb, int(60e3), max_required_loss=1e-2, rtol=0.01, fraction=6, reg_param=1e-3, pde_param=1)
model.plot_history()

with open("hydrogen2.pickle", "wb") as f:
    dill.dump(model, f)
print("Saved")


n_plot = 100

grid1D = torch.linspace(-4., 4., n_plot)
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