import torch
from collections import OrderedDict

class DNN(torch.nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.depth = len(layers) - 1
        
        self.activation = activation
        
        l = []
        
        for i in range(self.depth - 1):
            l.append(
                (f"layer_{i}", torch.nn.Linear(layers[i], layers[i+1])))
            l.append(
                (f"activation_{i}", self.activation()))
        l.append(
            (f"layer_{self.depth-1}", torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(l)
        self.layers = torch.nn.Sequential(layerDict)
    
    def forward(self, x):
        out = self.layers(x)
        return out