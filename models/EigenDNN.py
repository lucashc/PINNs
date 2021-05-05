import torch
from collections import OrderedDict

class EigenDNN(torch.nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.depth = len(layers) - 1
        
        self.activation = activation
        
        # Build up network

        l = []

        # Eigenvalue layer
        self.Eig = torch.nn.Linear(1,1)

        # Rest of network
        for i in range(self.depth - 1):
            size = layers[i]
            if i == 0:
                size += 1
            l.append(
                (f"layer_{i}", torch.nn.Linear(size, layers[i+1])))
            l.append(
                (f"activation_{i}", self.activation()))
        l.append(
            (f"layer_{self.depth-1}", torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(l)
        self.layers = torch.nn.Sequential(layerDict)
    
    def forward(self, x):
        eigenvalue = self.Eig(torch.ones_like(x))
        out = self.layers(torch.cat((x, eigenvalue), 1))
        return out, eigenvalue