import pickle
import torch
from torch import nn
from functools import partial


def compute_F(y, y_old, h, k1, k2, k3):
    """[summary]

    Args:
        y ([type]): (batchsize, 3)
        y_old ([type]): (batchsize, 3)
        h ([type]): (batchsize,)
        k1 ([type]): (batchsize,)
        k2 ([type]): (batchsize,)
        k3 ([type]): (batchsize,)

    Returns:
        [type]: (batchsize, 3)
    """
    F = torch.zeros_like(y)
    F[:, 0] = y_old[:, 0] + h * (-k1 * y[:, 0] + k3 * y[:, 1] * y[:, 2]) - y[:, 0]
    F[:, 1] = y_old[:, 1] + h * (k1 * y[:, 0] - (k2 * y[:, 1] ** 2) - (k3 * y[:, 1] * y[:, 2])) - y[:, 1]
    F[:, 2] = y_old[:, 2] + h * (k2 * y[:, 1] ** 2) - y[:, 2]
    return F


def compute_J(y, h, k1, k2, k3):
    """[summary]

    Args:
        y ([type]): (batchsize, 3)
        h ([type]): (batchsize,)
        k1 ([type]): (batchsize,)
        k2 ([type]): (batchsize,)
        k3 ([type]): (batchsize,)

    Returns:
        [type]: (batchsize, 3, 3)
    """
    J = torch.zeros((len(h), 3, 3), device=y.device, dtype=y.dtype)
    J[:, 0, 0] = h * (-k1) - 1
    J[:, 0, 1] = h * k3 * y[:, 2]
    J[:, 0, 2] = h * k3 * y[:, 1]
    J[:, 1, 0] = h * k1
    J[:, 1, 1] = h * (-2 * k2 * y[:, 1] - k3 * y[:, 2]) - 1
    J[:, 1, 2] = h * (-k3 * y[:, 1])
    J[:, 2, 0] = 0
    J[:, 2, 1] = h * 2 * k2 * y[:, 1]
    J[:, 2, 2] = -1

    return J


class Newton(nn.Module):
    def __init__(self, tol, maxiter=100):
        """[summary]

        Args:
            tol ([type]): [description]
            maxiter (int, optional): [description]. Defaults to 100.
        """
        super().__init__()
        self.tol = tol
        self.maxiter = maxiter

    def forward(self, x, F, J, omega) -> torch.Tensor:
        """[summary]

        Args:
            x ([type]): initial guess. Shape = (batchsize, num_variables)
            F: function of x that returns (batchsize, num_variables)
            J: Jacobian of F that returns (batchsize, num_variables, num_variables)
            omeaga ([type]): relaxation factor

        Returns:
            torch.Tensor: Shape = (batchsize, num_variables)
        """
        _F = F(x)
        self.error = 1e9
        self.nit = 0
        while self.error > self.tol and self.nit < self.maxiter:
            _J = J(x)
            dx = torch.linalg.solve(_J, _F.unsqueeze(-1))
            x = x - omega * dx.squeeze()
            _F = F(x)
            self.error = torch.linalg.norm(_F)
            self.nit += 1
        # if self.error > self.tol:
        #     print("Warning: Newton solver did not converge")
        #     print("Error:", self.error)
        return x


class NewtonSOR(nn.Module):
    def __init__(self, tol, maxiter=100):
        """[summary]

        Args:
            tol ([type]): [description]
            omeaga ([type]): relaxation factor
            maxiter (int, optional): [description]. Defaults to 100.
        """
        super().__init__()
        self.tol = tol
        self.maxiter = maxiter

    def forward(self, x, F, J, omega) -> torch.Tensor:
        """[summary]

        Args:
            x ([type]): initial guess. Shape = (batchsize, num_variables)
            F: function of x that returns (batchsize, num_variables)
            J: Jacobian of F that returns (batchsize, num_variables, num_variables)
            omega: Shape = (batchsize, 1)

        Returns:
            torch.Tensor: Shape = (batchsize, num_variables)
        """
        _F = F(x)
        self.error = 1e9
        self.nit = 0
        while self.error > self.tol and self.nit < self.maxiter:
            _J = J(x)
            _D = torch.diag_embed(_J.diagonal(dim1=-2, dim2=-1))
            _L = -_J.tril(diagonal=-1)
            # dx, _ = torch.triangular_solve(_F.unsqueeze(-1), _D - omega.unsqueeze(-1) * _L, upper=False)
            dx = torch.linalg.solve_triangular(_D - omega.unsqueeze(-1) * _L, _F.unsqueeze(-1), upper=False)
            x = x - omega * dx.squeeze()
            _F = F(x)
            self.error = torch.linalg.norm(_F)
            self.nit += 1
        return x


class Indicator(nn.Module):
    def __init__(self, threshold, slope=1):
        super().__init__()
        self.threshold = threshold
        self.slope = slope

    def forward(self, x):

        return (
            (x > self.threshold).float()
            - torch.sigmoid(torch.relu(self.slope * (x - self.threshold))).detach()
            + torch.sigmoid(torch.relu(self.slope * (x - self.threshold)))
        )


# class Indicator(nn.Module):
#     def __init__(self, threshold, slope=1):
#         super().__init__()
#         self.threshold = threshold
#         self.slope = slope

#     def forward(self, x):

#         return (x > self.threshold).float() - torch.relu(x - self.threshold).detach() + torch.relu(x - self.threshold)


@torch.jit.script
def onesetp(x, _F, _J, omega):
    """[summary]

    Args:
        x ([type]): [description]
        _F ([type]): [description]
        _J ([type]): [description]

    Returns:
        [type]: [description]
    """
    _D = torch.diag_embed(_J.diagonal(dim1=-2, dim2=-1))
    _L = -_J.tril(diagonal=-1)
    dx = torch.linalg.solve_triangular(_D - omega.unsqueeze(-1) * _L, _F.unsqueeze(-1), upper=False)
    x = x - omega * dx.squeeze()
    return x


class NewtonSOR2(nn.Module):
    def __init__(self, tol, maxiter=100, slope=1):
        """[summary]

        Args:
            tol ([type]): [description]
            omeaga ([type]): relaxation factor
            maxiter (int, optional): [description]. Defaults to 100.
        """
        super().__init__()
        self.tol = tol
        self.indicator = torch.jit.script(Indicator(tol, slope))
        self.maxiter = maxiter
        self.F = None
        self.J = None

    def forward(self, x, F, J, omega) -> torch.Tensor:
        """[summary]

        Args:
            x ([type]): initial guess. Shape = (batchsize, num_variables)
            F: function of x that returns (batchsize, num_variables)
            J: Jacobian of F that returns (batchsize, num_variables, num_variables)
            omega: Shape = (batchsize, 1)

        Returns:
            torch.Tensor: Shape = (batchsize, num_variables)
        """
        _F = F(x)
        self.error = torch.linalg.norm(_F, axis=1)
        self.nit = torch.zeros_like(self.error)
        while any(self.error > self.tol) and all(self.nit < self.maxiter):
            _J = J(x)
            _D = torch.diag_embed(_J.diagonal(dim1=-2, dim2=-1))
            _L = -_J.tril(diagonal=-1)
            # dx, _ = torch.triangular_solve(_F.unsqueeze(-1), _D - omega.unsqueeze(-1) * _L, upper=False)
            dx = torch.linalg.solve_triangular(_D - omega.unsqueeze(-1) * _L, _F.unsqueeze(-1), upper=False)
            x = x - omega * dx.squeeze()
            _F = F(x)
            self.error = torch.linalg.norm(_F, axis=1)
            self.nit += self.indicator(self.error)
            # self.nit += torch.sigmoid(self.error - self.tol)
        return x


def clamp(x, min, max):
    out = torch.clamp(x, min, max).detach() - x.detach() + x
    return out


class NewtonSORJit(nn.Module):
    def __init__(
        self,
        tol,
        maxiter=100,
        slope=1,
        log=False,
        clamp_range=[-1.0, 2.0],
        grad_clamp=False,
        last_res=False,
        restart=False,
    ):
        """[summary]

        Args:
            tol ([type]): [description]
            omeaga ([type]): relaxation factor
            maxiter (int, optional): [description]. Defaults to 100.
        """
        super().__init__()
        self.tol = tol
        self.log = log
        self.grad_clamp = grad_clamp
        if self.log:
            self.indicator = Indicator(torch.log(torch.tensor(tol)), torch.tensor(slope))
        else:
            self.indicator = Indicator(torch.tensor(tol), torch.tensor(slope))
        self.maxiter = maxiter
        self.error = torch.empty(1)
        self.error_hist = torch.empty(1)
        self.nit = torch.empty(1)
        self.min = torch.tensor(clamp_range[0])
        self.max = torch.tensor(clamp_range[1])
        self.eps = torch.finfo(torch.float64).eps
        self.last_res = last_res
        self.restart = restart

    def forward(self, x, omega, y=None) -> torch.Tensor:
        """[summary]

        Args:
            x ([type]): initial guess. Shape = (batchsize, num_variables)
            F: function of x that returns (batchsize, num_variables)
            J: Jacobian of F that returns (batchsize, num_variables, num_variables)
            omega: Shape = (batchsize, 1)

        Returns:
            torch.Tensor: Shape = (batchsize, num_variables)
        """
        y_old = x[:, :3].clone()
        h = x[:, 3].clone()
        k1 = x[:, 4].clone()
        k2 = x[:, 5].clone()
        k3 = x[:, 6].clone()
        if y is None:
            y = x[:, :3].clone()
        device = y_old.device
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        if self.last_res:
            self.error_hist = torch.empty(h.shape[0], self.maxiter + 1, device=device)
        self.nit = torch.zeros_like(h)
        _F = compute_F(y=y, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)
        self.error = torch.linalg.norm(_F, dim=1)
        i = 0
        if self.last_res:
            self.error_hist[:, i] = self.error

        while any(self.error > self.tol) and i < self.maxiter:

            if self.log:
                self.nit += self.indicator(torch.log(self.eps + self.error))
            else:
                self.nit += self.indicator(self.error)
            i += 1

            _J = compute_J(y=y, h=h, k1=k1, k2=k2, k3=k3)
            _D = torch.diag_embed(_J.diagonal(dim1=-2, dim2=-1))
            _L = -_J.tril(diagonal=-1)
            # dx, _ = torch.triangular_solve(_F.unsqueeze(-1), _D - omega.unsqueeze(-1) * _L, upper=False)
            dx = torch.linalg.solve_triangular(_D - omega.unsqueeze(-1) * _L, _F.unsqueeze(-1), upper=False)
            y = y - omega * dx.squeeze()
            if self.grad_clamp:
                y = clamp(y, min=self.min, max=self.max)
            else:
                y = torch.clamp(y, min=self.min, max=self.max)
            # y.register_hook(print_grad)  # for debugging

            _F = compute_F(y=y, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)
            self.error = torch.linalg.norm(_F, dim=1)
            if self.last_res:
                self.error_hist[:, i] = self.error.clone()
            if self.restart:
                if y.max() > 1.5 or y.min() < -0.5:
                    return torch.tensor(False)

        # i = 0
        # self.error_hist[:, i] = self.error.clone()
        # while self.error.max() > self.tol and i < self.maxiter:
        #     if self.log:
        #         self.nit += self.indicator(torch.log(self.eps + self.error))
        #     else:
        #         self.nit += self.indicator(self.error)
        #     i += 1

        #     _J = compute_J(y=y, h=h, k1=k1, k2=k2, k3=k3)
        #     _D = torch.diag_embed(_J.diagonal(dim1=-2, dim2=-1))
        #     _L = -_J.tril(diagonal=-1)
        #     # dx, _ = torch.triangular_solve(_F.unsqueeze(-1), _D - omega.unsqueeze(-1) * _L, upper=False)
        #     dx = torch.linalg.solve_triangular(_D - omega.unsqueeze(-1) * _L, _F.unsqueeze(-1), upper=False)
        #     y = y - omega * dx.squeeze()
        #     if self.grad_clamp:
        #         y = clamp(y, min=self.min, max=self.max)
        #     else:
        #         y = torch.clamp(y, min=self.min, max=self.max)
        #     # y.register_hook(print_grad)  # for debugging
        #     _F = compute_F(y=y, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)

        #     self.error = torch.linalg.norm(_F, dim=1)
        #     self.error_hist[:, i] = self.error.clone()

        return y


def print_grad(grad):
    print(grad, grad.shape, grad.isnan().sum())


def solve_robertson(
    k1, k2, k3, y_init, omega, hs, tol, maxiter, method, meta_learner=None, save_dir=None, zero_initial_guess=False
):
    """[summary]

    Args:
        k1 ([type]): (batchsize,)
        k2 ([type]): (batchsize,)
        k3 ([type]): (batchsize,)
        y_init ([type]): (batchsize, 3)
        omega ([type]): (batchsize, 1)
        hs ([type]): (batchsize, num_steps)
        tol ([type]): float
        method ([type]): [description]
        save_dir ([type], optional): [description]. Defaults to None.
        gbms: Defaults to None.

    Returns:
        Y: (batchsize, 3, num_steps + 1)
        nit_hist: (num_steps)
        omega_hist: (num_steps)
    """

    if method == "newton":
        solver = Newton(tol, maxiter)
    elif method == "newton_sor":
        solver = NewtonSOR(tol, maxiter)
    elif method == "newton_sor_jit":
        solver = torch.jit.script(NewtonSORJit(tol, maxiter, restart=True))

    # initialize
    batchsize = hs.shape[0]
    num_steps = hs.shape[1]
    Y = torch.zeros((batchsize, 3, num_steps + 1))
    Y[:, :, 0] = y_init
    nit_hist = torch.zeros(num_steps)
    omega_hist = torch.zeros(num_steps)

    for i in range(num_steps):
        y_old = Y[:, :, i].clone()
        h = hs[:, i]
        F = partial(compute_F, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)
        J = partial(compute_J, h=h, k1=k1, k2=k2, k3=k3)
        x = torch.cat([y_old, h.unsqueeze(-1), k1.unsqueeze(-1), k2.unsqueeze(-1), k3.unsqueeze(-1)], axis=1)
        if meta_learner:
            omega, y0 = meta_learner(x.clone())
        else:
            y0 = y_old.clone()
        if zero_initial_guess:
            y = torch.zeros_like(y0)
        if method == "newton_sor_jit":
            y = torch.tensor(False)

            while not y.is_floating_point():
                y = solver(x.clone(), omega, y0.clone())
                nit_hist[i] += solver.nit.mean()
                if not y.is_floating_point():
                    omega *= 0.95
                    print("restart")

            Y[:, :, i + 1] = y

        else:
            y = solver(y_old, F, J, omega)
            Y[:, :, i + 1] = y
            nit_hist[i] = solver.nit
        omega_hist[i] = omega

        if save_dir:
            pickle.dump((x, y), open(save_dir + f"{i}.pkl", "wb"))

    return Y, nit_hist, omega_hist
