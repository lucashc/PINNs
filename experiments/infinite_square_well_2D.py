#%%
# import torch
import torch
import numpy as np
from scipy.constants import hbar, m_e
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
from collections import defaultdict
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if torch.cuda.is_available():
    if input("Use GPU? [y/n]")=="y":
        dev = "cuda"
    else:
        dev = "cpu"
else:
    print("GPU not availible to PyTorch")
    dev = "cpu"
device = torch.device(dev)

# Settings

L = 1 # m

# Divide by zero avoidance
eps = 1e-6

# Solution on bounds
x_min = 0
x_max = L

# Characteristic energy
E_char = 1 #hbar**2/m_e/L**2

class Sin(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.sin(x)


def dfx(x, f):
    return torch.autograd.grad([f], [x], grad_outputs=torch.ones_like(f), create_graph=True)[0]


def PDE_loss(x, psi, E):
    psi_dx = dfx(x, psi)
    psi_ddx = dfx(x, psi_dx[:,0])[:,0] + dfx(x, psi_dx[:,1])[:,1]
    diff = 1/2 * psi_ddx + E * psi # -hbar**2/(2*m_e) * psi_ddx + potential(x) * psi  - E * psi
    loss = torch.mean(diff**2)
    return loss

def compose_psi(x, N):
    f_b = 0
    dt = x - x_min
    psi = f_b + (1 - torch.exp(-dt[:, 0])) * (1 - torch.exp(-dt[:, 1])) * (1 - torch.exp(dt[:,0]-x_max)) * (1 - torch.exp(dt[:,1]-x_max)) * N.reshape(-1)
    return psi

def perturb(grid, x_min=0, x_max=1, n_train=101, sig=0.05):
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

def c_update(i):
    return 1. + i




class DNN2D(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()

        self.activation = Sin()

        self.Ein = torch.nn.Linear(1, 1, bias=False)
        self.Ein.weight.data.fill_(6.)
        self.Lin_1 = torch.nn.Linear(3, hidden_size)
        self.Lin_2 = torch.nn.Linear(hidden_size, hidden_size)
        # self.Lin_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.final = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Lambda
        eigenvalue = self.Ein(torch.ones_like(x[:,0].reshape(-1,1)))
        # Actual network
        L1 = self.Lin_1(torch.cat((x, eigenvalue), 1))
        h1 = self.activation(L1)
        L2 = self.Lin_2(h1)
        h2 = self.activation(L2)
        # L3 = self.Lin_2(h2)
        # h3 = self.activation(L3)
        out = self.final(h2).to(device)
        return out, eigenvalue

#%%
def train(hidden_size, epochs, n_train, lr, minibatches=1):
    # Network initalization
    network = DNN2D(hidden_size)
    network = network.to(device)
    betas = [0.999, 0.9999]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=betas)

    # Domain intialization
    grid1D = torch.linspace(x_min, x_max, n_train)
    grid2D_x, grid2D_y = torch.meshgrid(grid1D,grid1D)
    grid = torch.cat([grid2D_x.reshape(-1,1), grid2D_y.reshape(-1,1)], dim=1)

    # Storage of intermediate networks
    stub = lambda: (None, 1e20)
    storage = defaultdict(stub)

    # Histories
    En_history = []
    # Commented out as they take too much memory
    L_drive_history = []
    L_PDE_history = []
    L_f_history = []
    L_lambda_history = []
    #L_tot_history = []
    #c_history = []
    epoch_loss_history = []

    # Driver
    c = c_update(0)
    c_index = 0

    bar = tqdm(range(epochs), desc="Energy: ~; c: ~")

    for epoch in bar:

        x_train = perturb(grid, x_max, x_min, n_train, sig=0.2*(x_max-x_min) / (n_train-1))

        random_order = np.random.permutation(n_train*n_train)

        # Batch setup
        batch_size = n_train*n_train // minibatches
        batch_start, batch_end = 0, batch_size

        x_batch = x_train[random_order, :]
        x_batch.requires_grad = True

        epoch_loss = 0.0


        for n in range(minibatches):

            x_minibatch = x_batch[batch_start:batch_end, :]
            x_minibatch = x_minibatch.to(device)
            nn, En = network(x_minibatch)
            En_history.append(En[0].detach().cpu().numpy()[0])

            psi = compose_psi(x_minibatch, nn)
            L_PDE = PDE_loss(x_minibatch, psi, En)

            # Update driver
            if epoch % 1500 == 0:
                c_index += 1
                c = c_update(c_index)
            
            L_drive = torch.mean(torch.exp(-En + c))
            L_lambda = 1/(torch.mean(En**2) + eps)
            L_f = 1/(torch.mean(nn**2) + eps)

            L_reg = L_drive + L_lambda + L_f

            L_tot = 1e-3*L_reg + L_PDE

            # Log
            L_PDE_history.append(L_PDE.detach().cpu().numpy())
            L_f_history.append(L_f.detach().cpu().numpy())
            L_drive_history.append(L_drive.detach().cpu().numpy())
            L_lambda_history.append(L_lambda.detach().cpu().numpy())
            #L_tot_history.append(L_tot)
            #c_history.append(c)

            # Train step

            L_tot.backward(retain_graph=False)
            optimizer.step()
            epoch_loss += L_tot.detach().cpu().numpy()
            optimizer.zero_grad()

            batch_start += batch_size
            batch_end += batch_size
        
        
        E_bin = abs(En[0].detach().cpu().numpy()[0]//10)
        criterion = L_PDE.clone().detach().cpu().numpy()
        if criterion < storage[E_bin][1]:
            storage[E_bin] = (copy.deepcopy(network), criterion)

        
        epoch_loss_history.append(epoch_loss)
        bar.set_description(f"Energy: {En[0].detach().cpu().numpy()[0]:.4e}; c: {int(c):2d}")


    histories = {
        "En": En_history,
        "drive": L_drive_history,
        "PDE": L_PDE_history,
        "f": L_f_history,
        "lambda": L_lambda_history,
        #"tot": L_tot_history,
        #"c": c_history,
        "epoch": epoch_loss_history
    }
    return network, histories, storage

model, hist, storage = train(50, int(10e3), 11, 8e-3, 1)
#%%
plt.semilogy(hist["epoch"], label='epoch')
plt.semilogy(hist["PDE"], label='PDE')
plt.semilogy(hist["drive"], label='drive')
plt.semilogy(hist["f"], label='f')
plt.semilogy(hist["lambda"], label='lambda')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss over the epochs on log y scale")
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(hist["En"])
plt.xlabel("Epochs")
plt.ylabel("Eigenvalue")
plt.title("Change of eigenvalue over epochs")
plt.grid()
plt.show()
# %%
nn = storage[3][0]
X, Y = torch.meshgrid(torch.linspace(0,1,100),torch.linspace(0,1,100))
XY = torch.cat([X.reshape(-1,1), Y.reshape(-1,1)], dim=1)
fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
Z = compose_psi(XY.to(device), nn(XY.to(device))[0]).reshape(100,100)
# Z = compose_psi(XY, torch.ones_like(XY[:,0])).reshape(10,10)
surf = ax.plot_surface(X.detach().numpy(),
                       Y.detach().numpy(),
                       Z.detach().cpu().numpy(),
                       cmap=cm.coolwarm)
plt.show()
