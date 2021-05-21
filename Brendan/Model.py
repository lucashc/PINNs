#%% imports
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
dtype=torch.float
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

%matplotlib inline

#%% Functions and classes
class sinActivation(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)

def perturbPoints(grid, xL, xR, sig=0.5):
    dx = grid[1] - grid[0]
    noise = dx * torch.randn_like(grid) * sig
    x = grid + noise
    # x.data[2] = torch.ones(1,1)*(-1)
    x.data[x<xL] = xL - x.data[x<xL]
    x.data[x>xR] = 2*xR - x.data[x>xR]
    x.data[0] = torch.ones(1,1) * xL

    x.data[-1] = torch.ones(1,1) * xR
    x.requires_grad = False
    return x

def gradient(x, f):
    return torch.autograd.grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=torch.float), create_graph=True)[0]

def parametricSolution(x, nn, xL = 0, xR = 1, f0 = 0):
    N1, N2 = nn(x)
    g = (1 - torch.exp(-(x-xL)))*(1 - torch.exp(-(x-xR)))
    f = f0 + g * N1[0]
    return f

def hamiltonian_loss(x, f, E):
    f_dx = gradient(x, f)
    f_ddx= gradient(x, f_dx)
    hamiltonian = f_ddx/2 + E*f
    loss = (hamiltonian.pow(2)).mean()
    return loss

# def potential(x, V=40, L=1):
#     xnp = x.data.numpy()
#     pot = V * (np.heaviside(-xnp-L/2, 0.) + np.heaviside(xnp-L/2, 0.))
#     return torch.from_numpy(pot)

class PINN(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(PINN, self).__init__()
        self.activationFunction = sinActivation()

        # Define layers
        self.layerEnergy = torch.nn.Linear(1,1)
        self.layer1 = torch.nn.Linear(2, D_hid)
        self.layer2 = torch.nn.Linear(D_hid, D_hid)
        self.out = torch.nn.Linear(D_hid, 1)
    
    def forward(self, x):
        In1 = self.layerEnergy(torch.ones_like(x))
        L1 = self.layer1(torch.cat((x, In1), 1))
        h1 = self.activationFunction(L1)
        L2 = self.layer2(h1)
        h2 = self.activationFunction(L2)
        out = self.out(h2)
        return out, In1

#%% Training code
def run_scan(neurons, lr, xL=0, xR=1, n_train=100, epochs=20000):
    pinn = PINN(neurons)
    betas = [.999, .9999]
    c_drive = -4
    optimizer = optim.Adam(pinn.parameters(), lr=lr, betas=betas)
    Llim = 1e20
    E_history = []
    loss_history = []
    c_drive_history = []
    loss_H_history = []
    
    di = (None, 1e+20)
    dic = {0:di, 1:di, 2:di, 3:di, 4:di, 5:di, 6:di, 7:di, 8:di, 9:di, 10:di}

    grid = torch.linspace(xL, xR, n_train).reshape(-1,1)

    for t in tqdm(range(epochs)):
        grid_ = perturbPoints(grid, xL, xR, sig=0.03*xR)
        idx = np.random.permutation(n_train)
        x = grid_[idx]
        x.requires_grad = True
        loss = 0.

        nn, E = pinn(x)
        E_history.append(E[0].data.tolist()[0])

        f = parametricSolution(x, pinn, xL, xR, 0)
        # pot = potential(x)
        loss_tot = hamiltonian_loss(x, f, E) #, pot
        loss_H_history.append(loss_tot.data.numpy())
        
        if t%2500==0:
            c_drive += 1 
        c_drive_history.append(c_drive)
        loss_tot += 1/((f.pow(2)).mean()+1e-6) + 1/(E.pow(2).mean()+1e-6) + torch.exp(-1*E+c_drive).mean()
        # loss_tot = loss_H + loss_reg

        loss_tot.backward(retain_graph = False)
        optimizer.step()
        loss += loss_tot
        optimizer.zero_grad()

        loss_history.append(loss.data.numpy())
        
        if loss_tot < Llim:
            pinn_best = copy.deepcopy(pinn)
            Llim = loss_tot

        E_bin = abs(E[0].data.tolist()[0] // 10)
        if loss_tot < dic[E_bin][1]:
            dic[E_bin] = (copy.deepcopy(pinn), loss_tot)

    return pinn_best, E_history, loss_history, c_drive_history, loss_H_history, dic
# %% Training
np.random.seed(19961210)
pinn, E, loss, c_drive, loss_H, dic = run_scan(50, 8e-3, n_train=100, epochs=30000)
# %%
plt.plot(E)
plt.show()

plt.loglog(loss)
plt.show()

plt.plot(c_drive)
plt.show()

plt.loglog(loss_H)
plt.show()
# %%
grid = torch.linspace(0, 1, 100).reshape(-1,1)
grid.requires_grad = True
plt.plot(grid.detach().numpy(),parametricSolution(grid, dic[0][0]).detach().numpy())
plt.show()

plt.plot(grid.detach().numpy(),parametricSolution(grid, dic[1][0]).detach().numpy())
plt.show()

# plt.plot(grid.detach().numpy(),parametricSolution(grid, dic[2][0]).detach().numpy())
# plt.show()

# plt.plot(grid.detach().numpy(),parametricSolution(grid, dic[3][0]).detach().numpy())
# plt.show()

# %%
