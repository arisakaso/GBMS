#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from matplotlib import legend
from matplotlib.colors import LogNorm, Normalize
from src.data.data_module import Poisson1DDataModule
from src.models.train_poisson1d import GBMS1D
from src.utils.utils import get_config_and_ckpt, get_results, relative_l2_error1d

# %%
api = wandb.Api()
runs = api.sweep("sohei/poisson1d/bgi16tb7").runs

# %%
df = get_results(runs)
# %%
run = api.run("sohei/poisson1d/157fygl8")
df_base = get_results([run])
# %%
df_sol = df[df.solver == "jacobi"]
df_sol = df_sol[(df_sol.num_iter == 0) | (df_sol.num_iter == 64)]
df_sol = df_sol.sort_values("num_iter")

# %%
data_module = Poisson1DDataModule(
    data_path="/root/meta-pde-solver/data_share/raw/poisson1d_sin/100_True_False.pkl", num_data=10000, batch_size=1024
)
dl = data_module.test_dataloader()
import seaborn as sns

sns.set(context="talk")
import matplotlib

# del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

# fig, axes = plt.subplots(2,2)

j = 2477
err_list = []
for run_path in df_sol.run_path:
    conf, ckpt = get_config_and_ckpt(run_path)
    gbms = GBMS1D(conf).cuda()
    gbms = gbms.load_from_checkpoint(ckpt.name)
    gbms = gbms.eval()
    gbms = gbms.cuda().double()
    with torch.no_grad():
        for l, batch in enumerate(dl):
            if l == 0:
                continue
            x, y = batch
            x = x.cuda().double()
            y = y.cuda().double()
            f = x[:, [0]]
            dbc = x[:, [1]]
            y_hat = gbms.meta_learner(x)
            temp_list = []
            for k in range(65):
                if k in [0, 64]:
                    print(k)
                    plt.figure(figsize=(12, 8))
                    plt.plot(y[j].reshape(-1).cpu().numpy(), alpha=0.8, label="$u$")
                    plt.plot(y_hat[j].reshape(-1).cpu().numpy(), alpha=0.8, label="$\hat u$")
                    plt.ylim(-0.2, 0.2)
                    plt.legend()
                    plt.savefig(f"sample_{run_path}_{k}.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
                y_hat = gbms.jacobi(f, dbc, 1, y_hat)
            err = torch.mean((y - y_hat) ** 2, axis=[1, 2]).cpu().numpy()

    err_list.append(err)

# %%
(err_list[0] - err_list[-1]).argsort()
# %%
