import torch
from .EigenDNN import EigenDNN
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



class EigenvalueProblemModel:
    def __init__(self, layers, activation, composition, PDE_loss, lr=8e-3, betas=[0.999, 0.9999]):
        self.dnn = EigenDNN(layers, activation)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=lr, betas=betas)
        self.composition = composition
        self.PDE_loss = PDE_loss
    
    def train(self, driver, grid, perturb, epochs, n_train, minibatches=1):
        # Histories
        En_history = np.zeros(epochs*minibatches)
        epoch_loss_history = np.zeros(epochs)
        c = driver(0)

        bar = tqdm(range(epochs))

        for epoch in bar:
            X_train = perturb(grid)
            random_order = np.random.permutation(n_train)

            # Batch setup
            batch_size = n_train // minibatches
            batch_start, batch_end = 0, batch_size

            X_batch = X_train[random_order]
            X_batch.requires_grad = True

            epoch_loss = 0.0

            for n in range(minibatches):
                X_minibatch = X_batch[batch_start:batch_end]
                nn, En = self.dnn(X_minibatch)
                En_history[epoch*minibatches + n] = En[0].data.numpy()[0]

                psi = self.composition(X_minibatch, nn)
                L_PDE = self.PDE_loss(X_minibatch, psi, En)

                # Update driver
                c = driver(epoch+1)

                L_drive = torch.mean(torch.exp(-En+c))
                L_lambda = 1/(torch.mean(En**2))
                L_f = 1/(torch.mean(nn**2))

                L_reg = L_drive + L_lambda + L_f

                L_tot = L_reg + L_PDE

                # Train
                L_tot.backward(retain_graph=False)
                self.optimizer.step()
                epoch_loss += L_tot.data.numpy()
                self.optimizer.zero_grad()

                batch_start += batch_size
                batch_end += batch_size
            
            epoch_loss_history[epoch] = epoch_loss
            bar.set_description(f"Loss: {epoch_loss:.4e}")
    
        self.histories = {
            "En": En_history,
            "epoch": epoch_loss_history
        }
    
    def plot_history(self):
        plt.plot(self.histories["epoch"])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Loss over the epochs on log scale")
        plt.grid()
        plt.show()
        plt.plot(self.histories["En"])
        plt.xlabel("Epochs")
        plt.ylabel("Eigenvalue")
        plt.title("Change of eigenvalue over epochs")
        plt.grid()
        plt.show()


