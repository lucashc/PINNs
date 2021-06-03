import torch
from .EigenDNN import EigenDNN, EigenDNNMultiDimensional
from .helper import driver_loss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
from sklearn.cluster import AgglomerativeClustering



class EigenvalueProblemModel:
    def __init__(self, layers, activation, composition, PDE_loss, lr=8e-3, betas=[0.999, 0.9999], start_eigenvalue=1.0, normalize=False, lower_bound=None, upper_bound=None):
        self.dims = layers[0]
        if self.dims == 1:
            self.dnn = EigenDNN(layers, activation, start_eigenvalue)
        else:
            self.dnn = EigenDNNMultiDimensional(layers, activation, start_eigenvalue)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=lr, betas=betas)
        self.composition = composition
        self.PDE_loss = PDE_loss
        self.En_history = None
        self.eigenvalue_parts_history = None
        self.epoch_loss_history = None
        # Format is: (Loss, En, Network, eigenvalue_parts)
        stub = lambda: (float('inf'), None, None, None)
        self.eigenfunctions = defaultdict(stub)
        self.dnn_history = None
        self.normalize = normalize
        if self.normalize:
            if lower_bound is None or upper_bound is None:
                raise ValueError("Upper and lower bound must be specified when using normalization")
            else:
                self.upper_bound = upper_bound
                self.lower_bound = lower_bound

    
    def detect(self, index, L_PDE, eigenvalue_parts, drive_step, max_required_loss, rtol, fraction):
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
                self.eigenfunctions[marker] = (L_PDE, En, copy.deepcopy(self.dnn), eigenvalue_parts)
    
    def normalize_input_if_needed(self, x):
        # Normalizes to range -1:1
        if self.normalize:
            return (x-self.lower_bound)/(self.upper_bound-self.lower_bound)*2-1
        else:
            return x
    
    def train(self, driver, drive_step, grid, perturb, epochs, minibatches=1, max_required_loss=1e-4, rtol=0.001, fraction=3, driver_loss=driver_loss, reg_param=1, pde_param=100):
        # Histories
        self.En_history = np.zeros(epochs*minibatches)
        if self.dims > 1:
            self.En_parts_history = np.zeros((epochs*minibatches, self.dims))
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
                X_minibatch_normalized_when_needed = self.normalize_input_if_needed(X_minibatch)
                nn, En, En_parts = self.dnn(X_minibatch_normalized_when_needed)
                self.En_history[epoch*minibatches + n] = En[0].data.numpy()[0]
                if self.dims > 1:
                    self.En_parts_history[epoch*minibatches + n] = En_parts[0].data.numpy()

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
            self.detect(epoch, L_PDE.data.numpy(), En_parts, drive_step, max_required_loss, rtol, fraction)
            bar.set_description(f"Loss: {self.L_PDE_history[epoch*minibatches]:.4e}; Eigenvalue: {self.En_history[epoch*minibatches]:.4e}; c: {c:.4e}")
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
        if self.dims > 1:
            for i in range(self.dims):
                plt.plot(self.En_parts_history[:, i], label=f"Dim {i+1}")
            plt.xlabel("Epochs")
            plt.ylabel("Eigenvalue")
            plt.legend()
            plt.title("Change of eigenvalue parts over epochs")
            plt.grid()
            plt.show()
    
    def get_eigenfunction(self, marker):
        dnn = self.eigenfunctions[marker][2]
        def wrapper(x):
            nn, _, _ = dnn(x)
            return self.composition(x, nn)
        return wrapper
    
    def remove_redundancies(self, threshold, grid):
        # Fix order
        functions = list(self.eigenfunctions.items())
        # Compute distance matrix
        dist_matrix = np.zeros((len(functions), len(functions)))
        for index1, func1 in enumerate(functions):
            for index2, func2 in enumerate(functions):
                dist_matrix[index1, index2] = torch.mean(torch.abs(torch.abs(func1[1][2](grid)[0])-torch.abs(func2[1][2](grid)[0])))

        # Excute Hierachical clustering
        clustering = AgglomerativeClustering(affinity="precomputed", n_clusters=None, distance_threshold=threshold, linkage="complete").fit(dist_matrix)
        labels = clustering.labels_

        print(f"Found {clustering.n_clusters_} clusters")

        # Form relevant clusters
        clusters = defaultdict(lambda: [])
        for index, label in enumerate(labels):
            clusters[label].append((functions[index]))
        
        # Reduce to only best
        pruned = {}
        for label in labels:
            best_in_cluster = sorted(clusters[label], key=lambda x: x[1][0])[0]
            pruned[best_in_cluster[0]] = best_in_cluster[1]
        print(f"Pruned {len(functions)-len(pruned)} of total functions {len(functions)}")
        return pruned



