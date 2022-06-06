import torch
from src.solvers.newton import Newton, NewtonSOR, NewtonSORJit
from functools import partial
import pickle


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


# def solve_robertson(k1, k2, k3, y_init, omega, hs, tol, maxiter, method, meta_learner=None, save_dir=None):
#     """[summary]

#     Args:
#         k1 ([type]): (batchsize,)
#         k2 ([type]): (batchsize,)
#         k3 ([type]): (batchsize,)
#         y_init ([type]): (batchsize, 3)
#         omega ([type]): (batchsize, 1)
#         hs ([type]): (batchsize, num_steps)
#         tol ([type]): float
#         method ([type]): [description]
#         save_dir ([type], optional): [description]. Defaults to None.
#         gbms: Defaults to None.

#     Returns:
#         Y: (batchsize, 3, num_steps + 1)
#         nit_hist: (num_steps)
#         omega_hist: (num_steps)
#     """

#     if method == "newton":
#         solver = Newton(tol, maxiter)
#     elif method == "newton_sor":
#         solver = NewtonSOR(tol, maxiter)
#     elif method == "newton_sor_jit":
#         solver = torch.jit.script(NewtonSORJit(tol, maxiter))

#     # initialize
#     batchsize = hs.shape[0]
#     num_steps = hs.shape[1]
#     Y = torch.zeros((batchsize, 3, num_steps + 1))
#     Y[:, :, 0] = y_init
#     nit_hist = torch.zeros(num_steps)
#     omega_hist = torch.zeros(num_steps)

#     for i in range(num_steps):
#         y_old = Y[:, :, i]
#         h = hs[:, i]
#         F = partial(compute_F, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)
#         J = partial(compute_J, h=h, k1=k1, k2=k2, k3=k3)
#         if meta_learner:
#             x = torch.cat([y_old, h.unsqueeze(-1), k1.unsqueeze(-1), k2.unsqueeze(-1), k3.unsqueeze(-1)], axis=1)
#             omega = meta_learner(x)
#         if method == "newton_sor_jit":
#             Y[:, :, i + 1] = solver(y_old, omega)
#             nit_hist[i] = solver.nit.mean()
#         else:
#             Y[:, :, i + 1] = solver(y_old, F, J, omega)
#             nit_hist[i] = solver.nit
#         omega_hist[i] = omega

#     return Y, nit_hist, omega_hist


def solve_robertson(k1, k2, k3, y_init, omega, hs, tol, maxiter, method, meta_learner=None, save_dir=None):
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
        solver = torch.jit.script(NewtonSORJit(tol, maxiter))

    # initialize
    batchsize = hs.shape[0]
    num_steps = hs.shape[1]
    Y = torch.zeros((batchsize, 3, num_steps + 1))
    Y[:, :, 0] = y_init
    nit_hist = torch.zeros(num_steps)
    omega_hist = torch.zeros(num_steps)

    for i in range(num_steps):
        y_old = Y[:, :, i]
        h = hs[:, i]
        F = partial(compute_F, y_old=y_old, h=h, k1=k1, k2=k2, k3=k3)
        J = partial(compute_J, h=h, k1=k1, k2=k2, k3=k3)
        x = torch.cat([y_old, h.unsqueeze(-1), k1.unsqueeze(-1), k2.unsqueeze(-1), k3.unsqueeze(-1)], axis=1)
        if meta_learner:
            omega, y = meta_learner(x.clone())
        if method == "newton_sor_jit":
            y = solver(x, omega, y_old)
            Y[:, :, i + 1] = y
            nit_hist[i] = solver.nit.mean()
        else:
            y = solver(y_old, F, J, omega)
            Y[:, :, i + 1] = y
            nit_hist[i] = solver.nit
        omega_hist[i] = omega

        if save_dir:
            pickle.dump((x, y), open(save_dir + f"{i}.pkl", "wb"))

    return Y, nit_hist, omega_hist
