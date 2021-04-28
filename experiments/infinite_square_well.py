import torch
import numpy as np
from scipy.constants import hbar, m_e
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    return torch.autograd.grad([f], [x], grad_outputs=torch.ones_like(x), create_graph=True)[0]


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

def perturb(grid, sig=0.5):
    delta_x = grid[1] - grid[0]
    noise = delta_x * torch.randn_like(grid) * sig
    x = grid + noise
    # Make sure perturbation still lay in domain
    x[x < x_min] = x_min - x[x < x_min]
    x[x > x_max] = 2*x_max - x[x > x_max]
    # Make sure at least one point is at the boundaries
    x[0] = torch.ones(1,1)*x_min
    x[-1] = torch.ones(1,1)*x_max

    return x




class DNN(torch.nn.Module):
    def __init__(self, hidden_size=10):
        super().__init__()

        self.activation = Sin()

        self.Ein = torch.nn.Linear(1, 1)
        self.Lin_1 = torch.nn.Linear(2, hidden_size)
        self.Lin_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.final = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Lambda
        eigenvalue = self.Ein(torch.ones_like(x))
        # Actual network
        L1 = self.Lin_1(torch.cat((x, eigenvalue), 1))
        h1 = self.activation(L1)
        L2 = self.Lin_2(h1)
        h2 = self.activation(L2)
        out = self.final(h2)
        return out, eigenvalue

def train(hidden_size, epochs, n_train, lr, minibatches=1):
    # Network initalization
    network = DNN(hidden_size)
    betas = [0.999, 0.9999]
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, betas=betas)

    # Domain intialization
    grid = torch.linspace(x_min, x_max, n_train).reshape(-1, 1)

    # Histories
    En_history = []
    #L_drive_history = []
    #L_PDE_history = []
    #L_f_history = []
    #L_lambda_history = []
    #L_tot_history = []
    #c_history = []
    epoch_loss_history = []

    # Driver
    c = -4*E_char

    bar = tqdm(range(epochs), desc="Loss: ~")

    for epoch in bar:

        x_train = perturb(grid)

        random_order = np.random.permutation(n_train)

        # Batch setup
        batch_size = n_train // minibatches
        batch_start, batch_end = 0, batch_size

        x_batch = x_train[random_order]
        x_batch.requires_grad = True

        epoch_loss = 0.0


        for n in range(minibatches):

            x_minibatch = x_batch[batch_start:batch_end]
            nn, En = network(x_minibatch)
            En_history.append(En[0].data.numpy()[0])

            psi = compose_psi(x_minibatch, nn)
            L_PDE = PDE_loss(x_minibatch, psi, En)

            # Update driver
            if epoch % 2500 == 0:
                c += E_char
            
            L_drive = torch.mean(torch.exp(-En + c))
            L_lambda = 1/(torch.mean(En**2) + eps)
            L_f = 1/(torch.mean(nn**2) + eps)

            L_reg = L_drive + L_lambda + L_f

            L_tot = L_reg + L_PDE

            # Log
            #L_PDE_history.append(L_PDE)
            #L_f_history.append(L_f)
            #L_drive_history.append(L_drive)
            #L_lambda_history.append(L_lambda)
            #L_tot_history.append(L_tot)
            #c_history.append(c)

            # Train step

            L_tot.backward(retain_graph=False)
            optimizer.step()
            epoch_loss += L_tot.data.numpy()
            optimizer.zero_grad()

            batch_start += batch_size
            batch_end += batch_size
        
        epoch_loss_history.append(epoch_loss)
        bar.set_description(f"Loss: {epoch_loss:.4e}")


    histories = {
        "En": En_history,
        #"drive": L_drive_history,
        #"PDE": L_PDE_history,
        #"f": L_f_history,
        #"lambda": L_lambda_history,
        #"tot": L_tot_history,
        #"c": c_history,
        "epoch": epoch_loss_history
    }
    return network, histories


model, hist = train(50, int(125e3), 100, 8e-3, 1)

input()
plt.loglog(hist["epoch"])
plt.show()

plt.plot(hist["En"])
plt.show()
