#%%
from __future__ import annotations

import torch


def get_jacobi_kernel():
    jacobi_kernel = torch.zeros(1, 1, 3, 3)
    jacobi_kernel[0, 0, 0, 1] = 1 / 4
    jacobi_kernel[0, 0, 1, 0] = 1 / 4
    jacobi_kernel[0, 0, 1, 2] = 1 / 4
    jacobi_kernel[0, 0, 2, 1] = 1 / 4
    return jacobi_kernel


class FixedConv2d(torch.nn.Conv2d):
    def __init__(self, kernel) -> None:
        super().__init__(1, 1, 3, padding=1)
        self.weight = torch.nn.Parameter(kernel, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


# def update_DB(x, dbc):
#     """
#     x: (batch_size, 1, width, height)
#     dbc: (batch_size, 1, width, height)
#     return: (batch_size, 1, width, height)
#     """
#     if (dbc[:, :, :, -1] != 0).all():
#         x[:, :, :, -1] = 0  # dp/dx = 0 at x = 2
#     if (dbc[:, :, 0, :] != 0).all():
#         x[:, :, 0, :] = 0  # dp/dy = 0 at y = 0
#     if (dbc[:, :, :, 0] != 0).all():
#         x[:, :, :, 0] = 0  # dp/dx = 0 at x = 0
#     if (dbc[:, :, -1, :] != 0).all():
#         x[:, :, -1, :] = 0  # dp/dy = 0 at y = 2

#     return x


def update_DB(x, dbc):
    """
    x: (batch_size, 1, width, height)
    dbc: (batch_size, 1, width, height)
    return: (batch_size, 1, width, height)
    """
    x[:, :, :, -1] = 0

    return x


# def update_NB(x, nbc):
#     """
#     x: (batch_size, 1, width, height)
#     nbc: (batch_size, 1, width, height)
#     return: (batch_size, 1, width, height)
#     """
#     if (nbc[:, :, :, -1] != 0).all():
#         x[:, :, :, -1] = x[:, :, :, -2]  # dp/dx = 0 at x = 2
#     if (nbc[:, :, 0, :] != 0).all():
#         x[:, :, 0, :] = x[:, :, 1, :]  # dp/dy = 0 at y = 0
#     if (nbc[:, :, :, 0] != 0).all():
#         x[:, :, :, 0] = x[:, :, :, 1]  # dp/dx = 0 at x = 0
#     if (nbc[:, :, -1, :] != 0).all():
#         x[:, :, -1, :] = x[:, :, -2, :]  # dp/dy = 0 at y = 2

#     return x


def update_NB(x, nbc):
    """
    x: (batch_size, 1, width, height)
    nbc: (batch_size, 1, width, height)
    return: (batch_size, 1, width, height)
    """
    x[:, :, 0, :] = x[:, :, 1, :]  # dp/dy = 0 at y = 0
    x[:, :, :, 0] = x[:, :, :, 1]  # dp/dx = 0 at x = 0
    x[:, :, -1, :] = x[:, :, -2, :]  # dp/dy = 0 at y = 2

    return x


class Jacobi2dUpdate(torch.nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.fconv = FixedConv2d(get_jacobi_kernel())
        self.h = h

    def forward(self, f, dbc, nbc, u):
        """[summary]

        Args:
            f ([type]): (b, 1, w, h)
            dbc ([type]): (b, 1, w, h)
            nbc ([type]): (b, 1, w, h)
            u ([type]): (b, 1, w, h)

        Returns:
            [type]: (b, 1, w, h)
        """

        u = self.fconv(u) - (self.h ** 2) / 4.0 * f
        u = update_DB(u, dbc)
        u = update_NB(u, nbc)
        return u


class Jacobi2d(torch.nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.update = Jacobi2dUpdate(h)

    def forward(self, f, dbc, nbc, num_iter, u0):
        """[summary]

        Args:
            f ([type]): (b, 1, w, h)
            dbc ([type]): (b, 1, w, h)
            nbc ([type]): (b, 1, w, h)
            num_iter ([type]): [description]
            u0 ([type]): (b, 1, w, h)

        Returns:
            [type]: [description]
        """
        u = u0
        for _ in range(num_iter):
            u = self.update(f, dbc, nbc, u)
        return u
