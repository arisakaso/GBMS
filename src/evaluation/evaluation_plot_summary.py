#%%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas.core.frame import DataFrame
from src.utils.utils import relative_l2_error
from yaml import load

sns.set(context="talk")

load_dir = "/root/meta-pde-solver/data_share/paper_tol"
results = {}
warmup = 100
episodes = [30]
# models = ["Baseline_sor", "NN_None_20_0_sor", "NN_sor_20_5_sor", "NN_sor_20_25_sor"]  # , "NN_sor_10_125_sor"]
train_solver = "sor"
test_solver = "sor"
models = ["baseline", "sor_0", "sor_4", "sor_16", "sor_64"]
nits = [100000]

df_p = []
for episode in episodes:
    fs_gt, ps_gt, ks_gt, us_gt, vs_gt, _, _, _, _, _, _ = pickle.load(
        open(os.path.join(load_dir, f"{episode}_gt.pkl"), "rb")
    )

    for model in models:
        for nit in nits:
            exp_name = f"{episode}_{model}_{test_solver}_{nit}"
            try:
                fs, ps, ks, us, vs, u_ins, v_ins, rho, nu, nit, tol = pickle.load(
                    open(os.path.join(load_dir, f"{exp_name}.pkl"), "rb")
                )

                p_errs = [relative_l2_error(ps_gt[:, i], ps[:, i]).cpu().numpy() for i in range(warmup, 1000)]
                v_errs = [
                    relative_l2_error(torch.cat([us_gt[:, i], vs_gt[:, i]]), torch.cat([us[:, i], vs[:, i]]))
                    .cpu()
                    .numpy()
                    for i in range(warmup, 1000)
                ]

                df_p.append(
                    {
                        "episode": episode,
                        "model": model,
                        "nit": nit,
                        "train_solver": train_solver,
                        "test_solver": test_solver,
                        "p_err": np.mean(p_errs),
                        "v_err": np.mean(v_errs),
                        "ks": np.mean(ks),
                    }
                )
                print(exp_name, "is loaded")
            except:
                print(exp_name, "does not exist")
df_p = pd.DataFrame(df_p)

df_p["model_train_solver"] = df_p.model + "_" + df_p.train_solver
#%%


plt.figure(figsize=(20, 10))
sns.barplot(data=df_p, x="nit", y="v_err", hue="model_train_solver")
plt.yscale("log")
plt.title(f"nit = {nit}")
plt.show()

# %%
for nit in df_p.nit.unique():
    plt.figure(figsize=(20, 10))
    temp_df = df_p[df_p.nit == nit].copy()
    temp_df["model_train_solver"] = temp_df.model + "_" + temp_df.train_solver + "_" + temp_df.train_nit.astype(str)
    sns.barplot(data=temp_df, x="train_step", y="v_err", hue="model_train_solver_train_iter")
    plt.yscale("log")
    plt.title(f"nit = {nit}")
    plt.show()

# %%
