#%%
from __future__ import annotations

import torch
import torch.nn.functional as F

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


# %%
def get_sor_kernel():
    sor_kernel = torch.zeros(1, 1, 3, 3)
    sor_kernel[0, 0, 0, 1] = 1 / 4
    sor_kernel[0, 0, 1, 0] = 1 / 4
    sor_kernel[0, 0, 1, 2] = 1 / 4
    sor_kernel[0, 0, 2, 1] = 1 / 4
    sor_kernel[0, 0, 1, 1] = -1
    return sor_kernel


# %%


class FixedConv2d(torch.nn.Conv2d):
    def __init__(self, kernel, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = torch.nn.Parameter(kernel, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


#%%


class SOR2dUpdate(torch.nn.Module):
    def __init__(self, h, omega, learn_omega=True) -> None:
        super().__init__()
        self.fconv = FixedConv2d(get_sor_kernel(), stride=2, padding=0)
        self.omega = torch.nn.Parameter(torch.tensor(omega), requires_grad=learn_omega)
        self.h = h

    def forward(self, f, dbc, nbc, u):
        u[:, :, 1:-1:2, 1:-1:2] += self.omega * (
            self.fconv(u[:, :, :, :].clone()) - (self.h ** 2) / 4.0 * f[:, :, 1:-1:2, 1:-1:2]
        )
        u[:, :, 1:-1:2, 2::2] += self.omega * (
            self.fconv(u[:, :, :, 1:].clone()) - (self.h ** 2) / 4.0 * f[:, :, 1:-1:2, 2::2]
        )
        u[:, :, 2::2, 1:-1:2] += self.omega * (
            self.fconv(u[:, :, 1:, :].clone()) - (self.h ** 2) / 4.0 * f[:, :, 2::2, 1:-1:2]
        )
        u[:, :, 2::2, 2::2] += self.omega * (
            self.fconv(u[:, :, 1:, 1:].clone()) - (self.h ** 2) / 4.0 * f[:, :, 2::2, 2::2]
        )
        u = update_DB(u, dbc)
        u = update_NB(u, nbc)
        return u


#%%


class SOR2d(torch.nn.Module):
    def __init__(self, h, omega, learn_omega=True) -> None:
        super().__init__()
        self.update = SOR2dUpdate(h, omega, learn_omega)

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


# %%
