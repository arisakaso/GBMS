#%%
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from src.solvers.jacobi2d import Jacobi2d
from src.solvers.sor2d import SOR2d
from src.utils.utils import relative_l2_error
from tqdm import tqdm


def velocity_boundary(u, v, u_in, v_in):
    """[summary]

    Args:
        u ([type]): (bs,1,nx,ny)
        v ([type]): (bs,1,nx,ny)
        u_in ([type]): (bs,ny)
        v_in ([type]): (bs,ny)

    Returns:
        [type]: [description]
    """

    u[:, 0, :, 0] = u_in  # 左側壁
    u[:, :, :, -1] = u[:, :, :, -2]  # 右側壁
    u[:, :, 0, :] = 0  # 底面壁
    u[:, :, -1, :] = 0  # 上面壁
    v[:, 0, :, 0] = v_in
    v[:, :, :, -1] = v[:, :, :, -2]
    v[:, :, 0, :] = 0.0
    v[:, :, -1, :] = 0.0
    return u, v


def compute_velocity(u, v, p, dx, dy, rho, nu, dt, u_in, v_in):
    """Compute velocity field

    Args:
        u ([type]): (batchsize, 1, Nx, Ny)
        v ([type]): (batchsize, 1, Nx, Ny)
        p ([type]): (batchsize, 1, Nx, Ny)
        dx ([type]): scalar
        dy ([type]): scalar
        rho ([type]): scalar
        nu ([type]): scalar
        dt ([type]): scalar
        u_in ([type]): (batchsize, Ny)
        v_in ([type]): (batchsize, Ny)

    Returns:
        [type]: [description]
    """

    un = u.clone()
    vn = v.clone()

    u[:, :, 1:-1, 1:-1] = (
        un[:, :, 1:-1, 1:-1]
        - un[:, :, 1:-1, 1:-1] * dt / dx * (un[:, :, 1:-1, 1:-1] - un[:, :, 1:-1, 0:-2])
        - vn[:, :, 1:-1, 1:-1] * dt / dy * (un[:, :, 1:-1, 1:-1] - un[:, :, 0:-2, 1:-1])
        - dt / (2 * rho * dx) * (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, 0:-2])
        + nu
        * (
            dt / dx ** 2 * (un[:, :, 1:-1, 2:] - 2 * un[:, :, 1:-1, 1:-1] + un[:, :, 1:-1, 0:-2])
            + dt / dy ** 2 * (un[:, :, 2:, 1:-1] - 2 * un[:, :, 1:-1, 1:-1] + un[:, :, 0:-2, 1:-1])
        )
    )

    v[:, :, 1:-1, 1:-1] = (
        vn[:, :, 1:-1, 1:-1]
        - un[:, :, 1:-1, 1:-1] * dt / dx * (vn[:, :, 1:-1, 1:-1] - vn[:, :, 1:-1, 0:-2])
        - vn[:, :, 1:-1, 1:-1] * dt / dy * (vn[:, :, 1:-1, 1:-1] - vn[:, :, 0:-2, 1:-1])
        - dt / (2 * rho * dy) * (p[:, :, 2:, 1:-1] - p[:, :, 0:-2, 1:-1])
        + nu
        * (
            dt / dx ** 2 * (vn[:, :, 1:-1, 2:] - 2 * vn[:, :, 1:-1, 1:-1] + vn[:, :, 1:-1, 0:-2])
            + dt / dy ** 2 * (vn[:, :, 2:, 1:-1] - 2 * vn[:, :, 1:-1, 1:-1] + vn[:, :, 0:-2, 1:-1])
        )
    )
    u, v = velocity_boundary(u, v, u_in, v_in)

    return u, v


def build_up_RHS(rho, dt, u, v, dx, dy):
    """[summary]

    Args:
        rho ([type]): [description]
        dt ([type]): [description]
        u ([type]): (b, 1, w, h)
        v ([type]): (b, 1, w, h)
        dx ([type]): [description]
        dy ([type]): [description]

    Returns:
        f: (b, 1, w, h)
    """

    f = torch.zeros_like(u)

    f[:, :, 1:-1, 1:-1] = rho * (
        1
        / dt
        * ((u[:, :, 1:-1, 2:] - u[:, :, 1:-1, 0:-2]) / (2 * dx) + (v[:, :, 2:, 1:-1] - v[:, :, 0:-2, 1:-1]) / (2 * dy))
        - ((u[:, :, 1:-1, 2:] - u[:, :, 1:-1, 0:-2]) / (2 * dx)) ** 2
        - 2
        * ((u[:, :, 2:, 1:-1] - u[:, :, 0:-2, 1:-1]) / (2 * dy) * (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, 0:-2]) / (2 * dx))
        - ((v[:, :, 2:, 1:-1] - v[:, :, 0:-2, 1:-1]) / (2 * dy)) ** 2
    )

    return f


def solve_pressure_poisson(x, nit, tol, meta_pde_solver, solver):
    err = 1
    f = x[:, [0]]
    dbc = x[:, [1]]
    nbc = x[:, [2]]
    if meta_pde_solver:
        p = meta_pde_solver.meta_learner(
            x[:, : meta_pde_solver.hparams.in_channels].float()
        ).double()  # better to be meta_learner
    else:
        p = x[:, [3]]  # previous step pressure

    i = 0
    total_err = 0
    for i in range(1, nit + 1):
        pn = solver(f, dbc, nbc, 1, p.clone())
        err = relative_l2_error(p, pn)
        total_err += err / nit
        p = pn
        if err < tol or torch.isnan(err):
            break
    p_res = p - solver(f, dbc, nbc, 1, p.clone())
    return p, i, err, total_err, p_res


def solve_pressure_poisson_train(x, nit, tol, meta_pde_solver, solver):
    f = x[:, [0]]
    dbc = x[:, [1]]
    nbc = x[:, [2]]

    p = meta_pde_solver.meta_learner(x[:, : meta_pde_solver.hparams.in_channels])  # better to be meta_learner

    for i in range(1, nit + 1):
        p = solver(f, dbc, nbc, 1, p)
    return p, None, None, None, None


def prepare_x(fs, dbc, nbc, ps, p_ress=None, num_iter=25):
    """[summary]

    Args:
        fs ([type]): (b, t, w, h)
        dbc ([type]): [description]
        nbc ([type]): [description]
        ps ([type]): [description]

    Returns:
        [type]: [description]
    """
    # shape of x = (b, c, w, h)

    if p_ress == None:
        p_ress = torch.zeros_like(ps)

    x = torch.cat(
        [
            fs[:, [-1]],
            dbc.expand(fs.shape[0], 1, dbc.shape[-2], dbc.shape[-1]),
            nbc.expand(fs.shape[0], 1, nbc.shape[-2], nbc.shape[-1]),
            ps[:, [-1]],
            ps[:, [-1]] - ps[:, [-2]],
            fs[:, [-1]] - fs[:, [-2]],
            # p_ress[:, [-1]],
            # torch.ones_like(ps[:, [-1]]) * num_iter,
            # us[:, [-1]],
            # vs[:, [-1]],
            # us[:, [-1]] - us[:, [-2]],
            # vs[:, [-1]] - vs[:, [-2]],
            # u_ins[n].expand(128, 128).T.reshape(1, 1, nx, ny),
            # v_ins[n].expand(128, 128).T.reshape(1, 1, nx, ny),
        ],
        axis=1,
    )

    return x.clone().detach()


def generate_inflow():

    u_ins = torch.zeros((1, nt, ny))  # (bs, time, y-axis)
    v_ins = torch.zeros((1, nt, ny))  # (bs, time, y-axis)
    window = torch.sin(y * np.pi)
    scale_t = torch.sin(t * np.pi)
    a = np.random.uniform(0, 10)
    b = np.random.uniform(-5, 5)
    c = np.random.uniform(0, 5)
    d = np.random.uniform(0, 2)
    e = np.random.uniform(0, np.pi / 2)
    angle_t = torch.sin((c * t - d) * np.pi) * e
    for n in range(nt):
        base = (0.5 + 0.5 * torch.sin((a * y - b * t[n]) * np.pi)) * window
        u_ins[:, n, :] = base * torch.cos(angle_t[n]) * scale_t[n]
        v_ins[:, n, :] = base * torch.sin(angle_t[n]) * scale_t[n]

    return u_ins, v_ins


if __name__ == "__main__":
    save_dir = "/root/meta-pde-solver/data_share/raw/sor1e9"
    with torch.no_grad():

        for episode in range(20, 40):
            # SET PARAMS
            ## precision
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)

            ## physical params
            rho = 1
            nu = 0.01

            ## space discretiazation
            nx = 128
            ny = 128
            dx = 1 / (nx - 1)
            dy = 1 / (ny - 1)
            x = torch.linspace(0, 1, 128)
            y = torch.linspace(0, 1, 128)

            ## time discretization
            lt = 1
            dt = 0.001
            nt = int(lt / dt)
            t = torch.linspace(0, lt, nt)

            ## params for pressure Poisson equation
            nit = 100000
            tol = 1e-9
            # solver = Jacobi2d(h=dx)
            solver = SOR2d(h=dx, omega=1.9)

            # BCs
            ## pressure
            dbc = torch.zeros((1, 1, nx, ny))
            dbc[:, :, :, -1] = 1  # rigth edge
            nbc = torch.zeros((1, 1, nx, ny))
            nbc[:, :, 0, :] = 1
            nbc[:, :, -1, :] = 1
            nbc[:, :, :, 0] = 1

            ## velocity
            u_ins, v_ins = generate_inflow()

            # INITIALIZE
            fs = torch.tensor(())
            ps = torch.tensor(())
            us = torch.tensor(())
            vs = torch.tensor(())
            ks = []

            f = torch.zeros((1, 1, nx, ny))
            p = torch.zeros((1, 1, nx, ny))
            u = torch.zeros((1, 1, nx, ny))
            v = torch.zeros((1, 1, nx, ny))

            for n in tqdm(range(nt)):

                # given info
                u_in = u_ins[:, n]
                v_in = v_ins[:, n]

                f = build_up_RHS(rho, dt, u, v, dx, dy)
                fs = torch.cat((fs, f), axis=1)

                if n < 2:
                    x = torch.zeros((1, 14, dbc.shape[-2], dbc.shape[-1]))
                else:
                    x = prepare_x(fs, dbc, nbc, ps)

                p, k, p_err = solve_pressure_poisson(x, nit, tol, None, solver)
                ps = torch.cat((ps, p), axis=1)
                ks.append(k)
                tqdm.write(f"{n}, {k}, {p_err}")

                u, v = compute_velocity(u, v, p, dx, dy, rho, nu, dt, u_in, v_in)
                us = torch.cat((us, u), axis=1)
                vs = torch.cat((vs, v), axis=1)

                pickle.dump(
                    [f.cpu(), p.cpu(), u.cpu(), v.cpu(), u_in.cpu(), v_in.cpu()],
                    open(os.path.join(save_dir, f"{episode}_{n}.pkl"), "wb"),
                )

            pickle.dump(
                [fs, ps, ks, us, vs, u_ins, v_ins, rho, nu, nit, tol],
                open(os.path.join(save_dir, f"{episode}.pkl"), "wb"),
            )

# %%
