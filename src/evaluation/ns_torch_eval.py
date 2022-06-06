#%%
from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch

#%%
import wandb
from models.train_ns import MetaPDESolver2D
from src.solvers.jacobi2d import Jacobi2d
from src.solvers.ns_torch import build_up_RHS, compute_velocity, prepare_x, solve_pressure_poisson
from src.solvers.sor2d import SOR2d
from src.utils.utils import get_config_and_ckpt, get_results
from tqdm import tqdm

#%%


if __name__ == "__main__":
    api = wandb.Api()
    sweep_id = "13dzp3wo"
    sweep = api.sweep(f"sohei/multi-steps/{sweep_id}")
    df = get_results(sweep.runs)
    df = df.query("name == 'stoic-sweep-3'")
    # save_dir = f"/root/meta-pde-solver/data_share/{sweep_id}"
    save_dir = f"/root/meta-pde-solver/data_share/paper5"
    load_dir = "/root/meta-pde-solver/data_share/raw/sor1e9"
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # SET PARAMS
    ## eval params
    warmup = 100
    episodes = [30, 31, 32, 33, 34]  # , 32, 33, 34]
    # f"{episode}_{model}_{train_solver}_{train_step}_{train_nit}_{test_solver}_{test_nit}"
    # models = dict(
    #     zip(
    #         df.solver + "_" + df.train_step.astype(str).str.zfill(2) + "_" + df.num_iter.astype(str).str.zfill(3),
    #         df.run_path,
    #     )
    # )
    models = {
        "sor_0": "sohei/multi-steps/joobzpl3",
        "sor_4": "sohei/multi-steps/n9r7rzmd",
        "sor_16": "sohei/multi-steps/xbys6obd",
        "sor_64": "sohei/multi-steps/7pneho8g",
        "baseline": None,
    }
    test_solver = "sor"

    #%%
    ## precision
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    ## space discretiazation
    nx = 128
    ny = 128
    dx = 1 / (nx - 1)
    dy = 1 / (ny - 1)

    ## time discretization
    lt = 1
    dt = 0.001
    nt = int(lt / dt)
    t = torch.linspace(0, lt, nt)

    with torch.no_grad():

        for episode in episodes:
            # LOAD DATA
            fs, ps, ks, us, vs, u_ins, v_ins, rho, nu, _, _ = pickle.load(
                open(os.path.join(load_dir, f"{episode}.pkl"), "rb")
            )
            # SAVE REFERENCE
            pickle.dump(
                [fs, ps, ks, us, vs, u_ins, v_ins, rho, nu, _, _],
                open(os.path.join(save_dir, f"{episode}_gt.pkl"), "wb"),
            )

            for model, run_path in models.items():

                meta_pde_solver = None
                if run_path:
                    conf, ckpt = get_config_and_ckpt(run_path)
                    meta_pde_solver = MetaPDESolver2D(conf).cuda()
                    meta_pde_solver = meta_pde_solver.load_from_checkpoint(ckpt.name)
                    meta_pde_solver = meta_pde_solver.eval()
                    meta_pde_solver = meta_pde_solver.cuda().float()  # .double()

                ## params for pressure Poisson equation
                solvers = {"sor": SOR2d(h=dx, omega=1.5, learn_omega=False)}

                tol = 1e-11
                for nit in [0, 4, 16, 64]:  # , 125, 625]:

                    # BCs
                    ## pressure
                    dbc = torch.zeros((1, 1, nx, ny))
                    dbc[:, :, :, -1] = 1  # rigth edge
                    nbc = torch.zeros((1, 1, nx, ny))
                    nbc[:, :, 0, :] = 1
                    nbc[:, :, -1, :] = 1
                    nbc[:, :, :, 0] = 1

                    # INITIALIZE
                    fs = fs[:, :warmup]
                    ps = ps[:, :warmup]
                    p_ress = torch.ones_like(ps) * 1e-9
                    us = us[:, :warmup]
                    vs = vs[:, :warmup]
                    ks = ks[:warmup]

                    u = us[:, [-1]]
                    v = vs[:, [-1]]

                    start = time.time()
                    for n in tqdm(range(warmup, nt)):

                        # given info
                        u_in = u_ins[:, n]
                        v_in = v_ins[:, n]

                        f = build_up_RHS(rho, dt, u, v, dx, dy)
                        fs = torch.cat((fs, f), axis=1)

                        x = prepare_x(fs, dbc, nbc, ps, p_ress, nit).clone()
                        factor = (torch.sqrt(torch.sum(x[:, [0], :, :] ** 2, axis=[2, 3], keepdim=True))) / float(1e6)
                        x = x / factor
                        p, k, err, total_err, p_res = solve_pressure_poisson(
                            x, nit, tol, meta_pde_solver, solvers[test_solver]
                        )
                        p = p * factor
                        ps = torch.cat((ps, p), axis=1)
                        p_ress = torch.cat((p_ress, p_res), axis=1)
                        ks.append(k)
                        # tqdm.write(f"{episode}, {model}, {n}, {k}, {err}")

                        u, v = compute_velocity(u, v, p, dx, dy, rho, nu, dt, u_in, v_in)
                        us = torch.cat((us, u), axis=1)
                        vs = torch.cat((vs, v), axis=1)

                    end = time.time()
                    print(end - start)
                    # pickle.dump(
                    #     [fs, ps, ks, us, vs, u_ins, v_ins, rho, nu, nit, tol],
                    #     open(os.path.join(save_dir, f"{episode}_{model}_{test_solver}_{nit}.pkl"), "wb"),
                    # )

    # %%
