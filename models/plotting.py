import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import spacing
from skimage import measure
import plotly.graph_objects as go
import torch
torch.set_default_dtype(torch.float64)


def plot_iso_surface(x_min, x_max, func):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    n_points = 50
    grid1D = torch.linspace(x_min, x_max, n_points)
    grid_x, grid_y, grid_z = torch.meshgrid(grid1D, grid1D, grid1D)
    grid = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1, 1)], dim=1)
    data = func(grid).reshape(n_points, n_points, n_points).detach()
    max_val = torch.max(data)
    verts, faces, _, _ = measure.marching_cubes(data.numpy(), max_val/3, spacing=((x_max-x_min)/n_points,)*3)
    result = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="Spectral", lw=0)
    plt.show()

def plot_iso_surface_plotly(x_min, x_max, func):
    n_points = 100
    grid1D = torch.linspace(x_min, x_max, n_points)
    grid_x, grid_y, grid_z = torch.meshgrid(grid1D, grid1D, grid1D)
    grid = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1, 1)], dim=1)
    data = func(grid).reshape(n_points, n_points, n_points).detach()
    max_val = float(torch.max(data).numpy())
    print(max_val)
    fig = go.Figure(data=go.Isosurface(
        x=grid_x.flatten(),
        y=grid_y.flatten(),
        z=grid_z.flatten(),
        value=data.flatten(),
        isomax=max_val/2,
        caps={"x_show": False, "y_show": False, "z_show": False}
    ))
    fig.show()