#%%
import matplotlib.pyplot as plt
import wandb
import yaml
from src.data.data_module_ns import PressurePoisson2DDataModule
from src.data.poisson import get_A
from tqdm import tqdm
import torch
from src.models.train_model_ns import MetaPDESolver2D
from src.solvers.jacobi2d import Jacobi2D
import torch.nn.functional as F
import pickle
import pandas as pd

#%%
run_path = "sohei/pressure_poisson/39xer2pf"
api = wandb.Api()
run = api.run(run_path)
#%%
def get_config_and_ckpt(run):

    for fi in run.files():
        if "config" in fi.name:
            conf = fi
            conf = wandb.restore(conf.name, run_path=run_path)
            conf = yaml.safe_load(open(conf.name))
            del conf["wandb_version"]
            conf = {k: v["value"] for k, v in conf.items()}
        if "ckpt" in fi.name:
            ckpt = fi
            ckpt = wandb.restore(ckpt.name, run_path=run_path)
    return conf, ckpt


conf, ckpt = get_config_and_ckpt(run)

#%%
meta_pde_solver = MetaPDESolver2D(conf).cuda()
meta_pde_solver = meta_pde_solver.load_from_checkpoint(ckpt.name)
meta_pde_solver = meta_pde_solver.eval()

#%%
vanila_solver = meta_pde_solver.solver
test_paths = [
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_20.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_21.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_22.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_23.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_24.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_25.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_26.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_27.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_28.pkl",
    "/root/meta-pde-solver/data/processed/pre_pressure_ns_0.01_1e-06_29.pkl",
]
data_module = PressurePoisson2DDataModule(
    train_paths=test_paths[:1],
    val_paths=test_paths[:1],
    test_paths=test_paths,
    # num_data=None,
    batch_size=256,
)

test_loss = []
with torch.no_grad():
    for batch in data_module.test_dataloader():
        x, p = batch
        x = x.cuda()
        p = p.cuda()

        f = x[:, :1]  # to keep dim
        G = x[:, 1:2]
        b = x[:, 2:3]
        pp = x[:, 3:]

        p_hat = vanila_solver(f=f, Gdb=G, dbc=b, Gnb=G, num_iter=125, u0=pp)
        test_loss.append(F.mse_loss(p, pp))
        break

# %%
torch.mean(torch.stack(test_loss))
# %%
