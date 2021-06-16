import torch
# torch.set_default_dtype(torch.float64)
import numpy as np
from scipy.constants import m_e, m_p, hbar, e, epsilon_0, pi, physical_constants
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.helper import dfx, Sin
from models.model import EigenvalueProblemModel
from matplotlib.cm import coolwarm
from math import sqrt

a_0 = physical_constants['Bohr radius'][0]
m_rel = (m_e+m_p)/m_p
c = e*m_e*a_0*a_0/hbar/hbar

# Solution on bounds
x_min = -6
x_max = 6

def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx[:,0])[:,0] + dfx(x, psi_dx[:,1])[:,1] + dfx(x, psi_dx[:,2])[:,2]
    diff = .5 * m_rel * psi_ddx + (c*E + potential(x)) * psi # -hbar**2/(2*m_e) * psi_ddx + potential(x) * psi  - E * psi
    loss = torch.mean(diff**2)
    return loss

def potential(x):
    return 1. / torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)

def compose_psi(x, N):
    f_b = 0
    # dt = x - x_min
    # psi = f_b + (1 - torch.exp(-dt[:, 0])) * (1 - torch.exp(-dt[:, 1])) * (1 - torch.exp(dt[:,0]-x_max*2)) * (1 - torch.exp(dt[:,1]-x_max*2)) * N.reshape(-1)
    psi = f_b + (torch.exp(-x[:,0]**2 - x[:,1]**2 - x[:,2]**2) - np.exp(-6)) / (1-np.exp(-6)) * N.reshape(-1)
    return psi

def perturb(grid, x_min=-6, x_max=6, sig=0.4):
    noise = torch.randn_like(grid) * sig
    x = grid + noise
    # Make sure perturbation still lay in domain
    r_2 = x[:,0]**2+x[:,1]**2+x[:,2]**2
    idx = r_2 > x_max**2
    x[idx,0] = x_max**2 / r_2[idx] * x[idx,0]
    x[idx,1] = x_max**2 / r_2[idx] * x[idx,1]
    x[idx,2] = x_max**2 / r_2[idx] * x[idx,2]
    return x

def driver(index):
    return -13.6/4#+.25*(index//1000)

transformation = [lambda x: torch.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2),
                  lambda x: torch.arccos(x[:,2]/torch.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2)),
                  lambda x: torch.atan2(x[:,1], x[:,0])
                  ]

def show_grid(grid, perturbed = False):
    p = perturb(grid, sig=.4) if perturbed else grid
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p[:,0], p[:,1], p[:,2])
    plt.show()

# grid1D = torch.linspace(x_min, x_max, 5)
# grid3D_x, grid3D_y, grid3D_z = torch.meshgrid(grid1D, grid1D, grid1D)
# grid = torch.cat([grid3D_x.reshape(-1,1), grid3D_y.reshape(-1,1), grid3D_z.reshape(-1,1)], dim=1)
n_train = 6
R = torch.linspace(0.01, 4., n_train)
theta=torch.linspace(0, 2*np.pi, n_train)
phi=torch.linspace(0, pi, n_train)
pol = torch.outer(R, torch.sin(phi)).reshape(-1)
R0 = torch.outer(R, torch.cos(phi)).reshape(-1)
grid_x = torch.outer(pol, torch.cos(theta))
grid_y = torch.outer(pol, torch.sin(theta))
grid_z = torch.outer(R0, torch.ones_like(theta))
grid = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)], dim=1)

x = perturb(grid)
x.requires_grad = True
r = transformation[0](x)
psi = 1/sqrt(pi)*torch.exp(-r)

model = EigenvalueProblemModel([3, 15, 15, 15, 15, 1], torch.nn.Tanh, compose_psi, PDE_loss, lr=8e-3, betas=[0.99, 0.999], start_eigenvalue=[-13.6/4, 1., 0.], transformation=transformation, get_energy=lambda En, En_parts: En_parts[:,0])
model.train(driver, 2000, grid, perturb, int(60e3), max_required_loss=1e-2, rtol=0.01, fraction=6, reg_param=1e-3, pde_param=1)
model.plot_history()

n_plot = 100

R = torch.linspace(0.01,6, 100)
phi=torch.linspace(0,2*np.pi, 100)
x = torch.outer(R, torch.cos(phi))
y = torch.outer(R, torch.sin(phi))
grid = torch.cat([x.reshape(-1,1), y.reshape(-1,1), torch.zeros_like(x.reshape(-1,1))], dim=1)

marker = input("Plot eigenvalue, marker: ['q'] to quit   ")
while marker != 'q':
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    try:
        psi = model.get_eigenfunction(marker)
        Z = psi(grid)
        # Z = normalise2D(Z, dx, dy)
        surf = ax.plot_surface(x.reshape(n_plot,n_plot).detach().numpy(),
                                y.reshape(n_plot,n_plot).detach().numpy(),
                                Z.reshape(n_plot,n_plot).detach().numpy(),
                                cmap = coolwarm)
        plt.show()
    except TypeError:
        print("invalid input")
    finally:
        marker = input("Plot eigenvalue, marker: ['q'] to quit   ")