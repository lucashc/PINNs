import torch
from collections import OrderedDict

class EigenDNN(torch.nn.Module):
    def __init__(self, layers, activation, start_eigenvalue=1.0):
        super().__init__()
        self.depth = len(layers) - 1
        
        self.activation = activation
        
        # Build up network

        l = []

        # Eigenvalue layer
        self.Eig = torch.nn.Linear(1,1, bias = False)
        self.Eig.weight.data.fill_(start_eigenvalue)

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
        eigenvalue = self.Eig(torch.ones((x.shape[0], 1)))
        out = self.layers(torch.cat((x, eigenvalue), 1))
        return out, eigenvalue, None

class EigenDNNMultiDimensional(torch.nn.Module):
    def __init__(self, layers, activation, start_eigenvalue=1.0, transformations=None):
        super().__init__()
        self.dims = layers[0]
        self.transformations = transformations
        if isinstance(start_eigenvalue, float) | isinstance(start_eigenvalue, int) :
            start_eigenvalue = [start_eigenvalue / self.dims] * self.dims

        # Build up network
        layers[0] = 1
        self.submodules = torch.nn.ModuleList([EigenDNN(layers, activation, start_eigenvalue[d]) for d in range(self.dims)])
    
    def forward(self, x):
        out = 1
        eigenvalue = 0.
        eigenvalue_parts = []
        for d in range(self.dims):
            outx, Ex, _ = self.submodules[d]((self.transformations[d](x) if self.transformations != None else x[:,d]).reshape(-1,1))
            out = out*outx
            eigenvalue += Ex
            eigenvalue_parts.append(Ex)
        eigenvalue_parts = torch.cat(eigenvalue_parts, dim=1)
        return out, eigenvalue, eigenvalue_parts