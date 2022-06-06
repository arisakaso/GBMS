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
run_paths = [
    # "sohei/poisson1d/gzk61vq7",  # tanh0
    # "sohei/poisson1d/p96oqeto",  # tanh4
    # "sohei/poisson1d/r1uvdnhy",  # tanh16
    # "sohei/poisson1d/zlou8jlv",  # tanh64
    # "sohei/poisson1d/7aicumfa",  # tanh0
    # "sohei/poisson1d/veofqfhd",  # tanh4
    # "sohei/poisson1d/cloyotst",  # tanh16
    # "sohei/poisson1d/6zurjn95",  # tanh64
    # "sohei/poisson1d/r8lto3mc",  # sin0
    # "sohei/poisson1d/dm5qk6s7",  # sin4
    # "sohei/poisson1d/tl6b15o4",  # sin16
    "sohei/poisson1d/iss1eotr",  # sin64
]
# %%
data_module = Poisson1DDataModule(
    data_path="/root/meta-pde-solver/data_share/raw/poisson1d/sin_simple_20_20210915.pkl",
    num_data=10000,
    batch_size=1024,
)
dl = data_module.test_dataloader()
# %%
rl2_dict = {}
for run_path in run_paths:
    print(run_path)
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
            factor = (torch.sqrt(torch.sum(x[:, [0], :] ** 2, axis=2, keepdim=True))) / float(
                gbms.hparams.normalize_factor
            )
            x = x / factor
            f = x[:, [0]]
            dbc = x[:, [1]]
            y_hat = gbms.meta_learner(x)
            temp_list = []
            for i in range(10000):
                y_hat = gbms.sor(f, dbc, 1, y_hat)
                # temp_list.append(relative_l2_error1d(y, y_hat * factor).cpu().numpy())
                temp_list.append(torch.nn.functional.mse_loss(y, y_hat * factor).cpu().numpy())
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
        for i in range(10000):
            y_hat = gbms.sor(f, dbc, 1, y_hat)
            # temp_list.append(relative_l2_error1d(y, y_hat).cpu().numpy())
            temp_list.append(torch.nn.functional.mse_loss(y, y_hat).cpu().numpy())
        rl2_dict["baseline"].append(np.array(temp_list))
rl2_dict["baseline"] = np.array(rl2_dict["baseline"]).mean(axis=0)


# %%
sns.set(context="poster")
import matplotlib

# del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "cm"

plt.figure(figsize=(14, 10))
labels = [
    r"$\Psi_{\mathrm{NN}}$ trained with $\Phi_{\mathrm{SOR},0}$",
    r"$\Psi_{\mathrm{NN}}$ trained with $\Phi_{\mathrm{SOR},4}$",
    r"$\Psi_{\mathrm{NN}}$ trained with $\Phi_{\mathrm{SOR},16}$",
    r"$\Psi_{\mathrm{NN}}$ trained with $\Phi_{\mathrm{SOR},64}$",
    # "Trained with 256 iters",
    r"$\Psi_{\mathrm{BL}}$",
]
colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    # "tab:purple",
    "k",
]
linestyles = [
    "-",
    "-",
    "-",
    "-",
    # "-",
    "--",
]
for v, label, ls, c in zip(rl2_dict.values(), labels, linestyles, colors):
    plt.plot(v, label=label, linestyle=ls, color=c, alpha=0.8)
plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.axvline(x=4, color="k", linestyle=":", alpha=0.8)
plt.axvline(x=16, color="k", linestyle=":", alpha=0.8)
plt.axvline(x=64, color="k", linestyle=":", alpha=0.8)
# plt.axvline(x=256, color="k", linestyle=":", alpha=0.8)
plt.xlabel("number of iterations")
plt.ylabel("MSE")
# plt.title("Convergence comparison")
# plt.savefig("convergence.eps", bbox_inches="tight", pad_inches=0.05, )
plt.savefig("convergence.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
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
