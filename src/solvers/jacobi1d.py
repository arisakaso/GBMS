#%%
from __future__ import annotations

import torch


class FixedConv1d(torch.nn.Conv1d):
    def __init__(self, kernel, in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = torch.nn.Parameter(kernel, requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


class Jacobi1d(torch.nn.Module):
    def __init__(self, h) -> None:
        super().__init__()
        self.fconv = FixedConv1d(self.get_jacobi_kernel_1d(h))

    def get_jacobi_kernel_1d(self, h):
        jacobi_kernel = torch.zeros(1, 2, 3)
        jacobi_kernel[0, 0, 0] = 1 / 2
        jacobi_kernel[0, 0, 2] = 1 / 2
        jacobi_kernel[0, 1, 1] = -(h ** 2) / 2.0
        return jacobi_kernel

    def update_DB(self, u, dbc):
        u[:, :, 0] = dbc[:, :, 0]
        u[:, :, -1] = dbc[:, :, -1]
        return u

    def update(self, f, dbc, u):
        u = self.fconv(torch.cat([u, f], axis=1))
        u = self.update_DB(u, dbc)
        return u

    def forward(self, f, dbc, num_iter, u0):
        """[summary]

        Args:
            f ([type]): (b, 1, w)
            dbc ([type]): (b, 1, w)
            nbc ([type]): (b, 1, w)
            num_iter ([type]): N
            u0 ([type]): (b, 1, w)

        Returns:
            [type]: [description]
        """
        u = u0
        for _ in range(num_iter):
            u = self.update(f, dbc, u)
        return u


# %%

# %%
