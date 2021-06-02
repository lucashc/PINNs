import torch
from .EigenDNN import EigenDNN, EigenDNNMultiDimensional
from .helper import driver_loss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy



class EigenvalueProblemModel:
    def __init__(self, layers, activation, composition, PDE_loss, lr=8e-3, betas=[0.999, 0.9999], start_eigenvalue=1.0):
        self.dnn = EigenDNN(layers, activation, start_eigenvalue) if layers[0]==1 else EigenDNNMultiDimensional(layers, activation, start_eigenvalue)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=lr, betas=betas)
        self.composition = composition
        self.PDE_loss = PDE_loss
        self.En_history = None
        self.epoch_loss_history = None
        # Format is: (Loss, En, Network)
        stub = lambda: (float('inf'), None, None)
        self.eigenfunctions = defaultdict(stub)
        self.dnn_history = None
    
    def detect(self, index, L_PDE, drive_step, max_required_loss, rtol, fraction):
        En = self.En_history[index]
        # Empirically, rtol=0.001 and drive_step//3 seem like good factors
        previous_close = np.sum(np.isclose(self.En_history[index-drive_step:index], En, atol=0, rtol=rtol))
        if previous_close > drive_step//fraction:
            marker = f"{En:1.1e}"
            if L_PDE < max_required_loss and L_PDE < self.eigenfunctions[marker][0]:
                if self.eigenfunctions[marker][1] is None:
                    tqdm.write(f"Found new eigenfunction {marker} with energy {En} and loss {L_PDE}")
                else:
                    tqdm.write(f"    Detected better eigenfunction {marker} with energy {En} and loss {L_PDE}")
                self.eigenfunctions[marker] = (L_PDE, En, copy.deepcopy(self.dnn))
    
    def train(self, driver, drive_step, grid, perturb, epochs, minibatches=1, max_required_loss=1e-4, rtol=0.001, fraction=3, driver_loss=driver_loss, reg_param=1, pde_param=100):
        # Histories
        self.En_history = np.zeros(epochs*minibatches)
        self.L_PDE_history = np.zeros(epochs*minibatches)
        self.epoch_loss_history = np.zeros(epochs)
        self.dnn_history = []
        c = driver(0)

        bar = tqdm(range(epochs))
        n_train = grid.shape[0]

        for epoch in bar:
            X_train = perturb(grid)
            random_order = np.random.permutation(n_train)

            # Batch setup
            batch_size = n_train // minibatches
            batch_start, batch_end = 0, batch_size

            X_batch = X_train[random_order,:]
            X_batch.requires_grad = True

            epoch_loss = 0.0

            for n in range(minibatches):
                X_minibatch = X_batch[batch_start:batch_end, :]
                nn, En = self.dnn(X_minibatch)
                self.En_history[epoch*minibatches + n] = En[0].data.numpy()[0]

                psi = self.composition(X_minibatch, nn)
                L_PDE = self.PDE_loss(X_minibatch, psi, En)

                # Update driver
                c = driver(epoch+1)

                L_drive = driver_loss(En, c)
                L_lambda = 1/(torch.mean(En**2)+1e-6)
                L_f = 1/(torch.mean(nn**2)+1e-6)

                L_reg = L_drive + L_lambda + L_f

                L_tot = reg_param*L_reg + pde_param*L_PDE

                self.L_PDE_history[epoch*minibatches + n] = L_PDE.detach().numpy()

                # Train
                L_tot.backward(retain_graph=False)
                self.optimizer.step()
                epoch_loss += L_tot.data.numpy()
                self.optimizer.zero_grad()

                batch_start += batch_size
                batch_end += batch_size
            
            self.epoch_loss_history[epoch] = epoch_loss
            self.detect(epoch, L_PDE.data.numpy(), drive_step, max_required_loss, rtol, fraction)
            bar.set_description(f"Loss: {self.L_PDE_history[epoch*minibatches]:.4e}; Eigenvalue: {self.En_history[epoch*minibatches]:.4e}; c: {c:.3}")
            if epoch%1000==0:
                self.dnn_history.append(copy.deepcopy(self.dnn))
    
    def plot_history(self):
        plt.plot(self.epoch_loss_history, label="Epoch loss")
        plt.xlabel("Epochs")
        plt.plot(self.L_PDE_history, label="PDE loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Loss on log scale")
        plt.grid()
        plt.legend()
        plt.show()
        plt.plot(self.En_history)
        plt.xlabel("Epochs")
        plt.ylabel("Eigenvalue")
        plt.title("Change of eigenvalue over epochs")
        plt.grid()
        plt.show()
    
    def get_eigenfunction(self, marker):
        dnn = self.eigenfunctions[marker][2]
        def wrapper(x):
            nn, _ = dnn(x)
            return self.composition(x, nn)
        return wrapper



