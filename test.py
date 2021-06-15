import torch
torch.set_default_dtype(torch.float64)
from models.plotting import plot_iso_surface_plotly
import dill
from scipy.constants import value

bohr_radius = value("Bohr radius")


with open("hydrogen2.pickle", 'rb') as f:
    model = dill.load(f)


def get_dnn_history_eigenfunction(model, index):
    dnn = model.dnn_history[index]
    def wrapper(x):
        nn, _, _ = dnn(x)
        return model.composition(x, nn)
    return wrapper

eigenfunction = model.get_eigenfunction("-1.4e+01")
#eigenfunction = get_dnn_history_eigenfunction(model, 40)

plot_iso_surface_plotly(-bohr_radius*20, bohr_radius*20, eigenfunction)