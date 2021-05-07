import torch

class Sin(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.sin(x)


def dfx(x, f):
    return torch.autograd.grad([f], [x], grad_outputs=torch.ones_like(x), create_graph=True)[0]


def perturb1D(grid, x_min, x_max, sig=0.5):
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


def perturb2D(grid_xx, grid_yy, x_min, x_max, y_min, y_max, sig=0.5):
    xx = perturb1D(grid_xx.flatten(), x_min, x_max, sig).reshape(grid_xx.shape)
    yy = perturb1D(grid_yy.flatten(), y_min, y_max, sig).reshape(grid_yy.shape)
    return xx, yy