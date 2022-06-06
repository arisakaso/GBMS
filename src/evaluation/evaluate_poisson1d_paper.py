#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import legend
from src.data.data_module import Poisson1DDataModule
from src.models.train_poisson1d import GBMS1D
from src.utils.utils import get_config_and_ckpt, get_results, relative_l2_error1d

# %%
api = wandb.Api()
runs = api.sweep("sohei/poisson1d/bgi16tb7").runs

# %%
df = get_results(runs)
# %%
df = df.sort_values(["solver", "num_iter"], ascending=[True, True])
df = df.dropna(axis=0)
run = api.run("sohei/poisson1d/157fygl8")
df_base = get_results([run])
# %%
columns = [
    "test_mse_jacobi_000",
    "test_mse_jacobi_004",
    "test_mse_jacobi_016",
    "test_mse_jacobi_064",
    # "test_rl2_jacobi_256",
    "test_mse_sor_000",
    "test_mse_sor_004",
    "test_mse_sor_016",
    "test_mse_sor_064",
    # "test_rl2_sor_256",
]
df_sol = df[df.num_iter <= 64]
# df_sol = df[df.solver == "sor"]
# df_sol = df
df_sol = pd.concat([df_sol, df_base])

import matplotlib.pyplot as plt

# %%
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

sns.set_context("talk")
plt.figure(figsize=(8, 6))
sns.heatmap(
    df_sol[columns],
    annot=True,
    fmt=".3f",
    norm=LogNorm(),
    xticklabels=["0", "4", "16", "64"],
    yticklabels=["baseline", "64", "16", "4", "0"],
)
plt.xlabel("test #iters")
plt.ylabel("training #iters")
plt.title("MSE")
# %%
columns = [
    "test_rl2_jacobi_000",
    "test_rl2_jacobi_004",
    "test_rl2_jacobi_016",
    "test_rl2_jacobi_064",
    # "test_rl2_jacobi_256",
    "test_rl2_sor_000",
    "test_rl2_sor_004",
    "test_rl2_sor_016",
    "test_rl2_sor_064",
    # "test_rl2_sor_256",
]
df_sol = df[df.num_iter <= 64]
# df_nn = df[(df.num_iter == 0) & (df.solver == "sor")]
df_sol = pd.concat([df_sol, df_base])


sns.set_context("talk")
# plt.figure(figsize=(8, 6))
ax = df_sol[columns].plot(kind="bar", figsize=(8, 6), legend=False)
ax.set_xticklabels(["baseline", "Jacobi 64 iters", "SOR 64 iters"], rotation=0)

plt.xlabel("Trained with")
plt.ylabel("Relative L2 error")
plt.title("Tested with Jacobi 64 iters")
# %%
df_sol = df[df.solver == "jacobi"]
# df_sol = df_sol[df_sol.num_iter <= 65]
df_sol = df_sol.sort_values("num_iter")
df_sol.num_iter
# %%
data_module = Poisson1DDataModule(
    data_path="/root/meta-pde-solver/data_share/raw/poisson1d_sin/100_True_False.pkl", num_data=10000, batch_size=1024
)
dl = data_module.test_dataloader()
# %%
rl2_dict = {}
for run_path in df_sol.run_path:
    conf, ckpt = get_config_and_ckpt(run_path)
    gbms = GBMS1D(conf).cuda()
    gbms = gbms.load_from_checkpoint(ckpt.name)
    gbms = gbms.eval()
    gbms = gbms.cuda().double()
    rl2_dict[run_path] = []
    with torch.no_grad():
        for batch in dl:
            x, y = batch
            x = x.cuda().double()
            y = y.cuda().double()
            f = x[:, [0]]
            dbc = x[:, [1]]
            y_hat = gbms.meta_learner(x)
            temp_list = []
            for i in range(1000):
                y_hat = gbms.jacobi(f, dbc, 1, y_hat)
                temp_list.append(relative_l2_error1d(y, y_hat).cpu().numpy())
            rl2_dict[run_path].append(np.array(temp_list))
    rl2_dict[run_path] = np.array(rl2_dict[run_path]).mean(axis=0)

#%%
rl2_dict["baseline"] = []
with torch.no_grad():
    for batch in dl:
        x, y = batch
        x = x.cuda().double()
        y = y.cuda().double()
        f = x[:, [0]]
        dbc = x[:, [1]]
        y_hat = x[:, [1]].clone()
        temp_list = []
        for i in range(1000):
            y_hat = gbms.jacobi(f, dbc, 1, y_hat)
            temp_list.append(relative_l2_error1d(y, y_hat).cpu().numpy())
        rl2_dict["baseline"].append(np.array(temp_list))
rl2_dict["baseline"] = np.array(rl2_dict["baseline"]).mean(axis=0)


# %%
sns.set(context="talk")
plt.figure(figsize=(14, 10))
labels = [
    "Trained with 0 iters",
    "Trained with 4 iters",
    "Trained with 16 iters",
    "Trained with 64 iters",
    "Trained with 256 iters",
    "Baseline",
]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "k"]
linestyles = ["-", "-", "-", "-", "-", "--"]
for v, label, ls, c in zip(rl2_dict.values(), labels, linestyles, colors):
    plt.plot(v, label=label, linestyle=ls, color=c)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.axvline(x=4, color="k", linestyle=":")
plt.axvline(x=16, color="k", linestyle=":")
plt.axvline(x=64, color="k", linestyle=":")
plt.axvline(x=256, color="k", linestyle=":")
plt.xlabel("test #iterations")
plt.ylabel("relative l2 error")
plt.title("Convergence comparison")

# %%
required_num = []
for v in rl2_dict.values():
    mask = v <= rl2_dict["sohei/poisson1d/uqbw0bno"][256]
    # mask = v <= 0.05
    required_num.append(mask.argmax())
required_num = sorted(required_num, reverse=True)
required_num
# %%
fig, ax = plt.subplots(figsize=(10, 8))
labels = [
    "Baseline",
    "0 ",
    "4 ",
    "16",
    "64",
    "256",
]
ax.bar(labels, required_num)
ax.set_xlabel("Trained with")
ax.set_ylabel("Requied #iters")
ax.set_title("Requied #iters to hit same acc as 256 iters")


# %%
df_sol = df[df.solver == "jacobi"]
# df_sol = df_sol[df_sol.num_iter <= 65]
df_sol = df_sol.sort_values("num_iter")
df_sol.num_iter
# %%
for i in range(300, 60000, 10):
    print(i)
    plt.plot(y[i].reshape(-1).cpu().numpy())
    plt.show()
    plt.gcf()
# %%
j = 1148
err_list = []
for run_path in df_sol.run_path:
    conf, ckpt = get_config_and_ckpt(run_path)
    gbms = GBMS1D(conf).cuda()
    gbms = gbms.load_from_checkpoint(ckpt.name)
    gbms = gbms.eval()
    gbms = gbms.cuda().double()
    with torch.no_grad():
        for batch in dl:
            x, y = batch
            x = x.cuda().double()
            y = y.cuda().double()
            f = x[:, [0]]
            dbc = x[:, [1]]
            y_hat = gbms.meta_learner(x)
            temp_list = []
            for i in range(257):
                if i in [0, 4, 16, 64, 256]:
                    print(i)
                    plt.figure(figsize=(12, 8))
                    plt.plot(y[j].reshape(-1).cpu().numpy(), alpha=0.8, label="u")
                    plt.plot(y_hat[j].reshape(-1).cpu().numpy(), alpha=0.8, label="u_hat")
                    plt.ylim(-0.2, 0.2)
                    plt.legend()
                    plt.show()
                y_hat = gbms.jacobi(f, dbc, 1, y_hat)
            err = torch.mean((y - y_hat) ** 2, axis=[1, 2]).cpu().numpy()
            break
    err_list.append(err)

# %%
(err_list[0] - err_list[-1]).argmax()
# %%
