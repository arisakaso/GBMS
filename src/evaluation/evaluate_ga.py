#%%
import pickle
from deap import base, creator, tools
from sympy import content

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 適応度クラスの作成
creator.create("Individual", list, fitness=creator.FitnessMin)  # 個体クラスの作成
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(context="talk")

path_baseline = [
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_1_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_0_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_2_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_3_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_4_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_5_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_6_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_7_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_8_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_baseline_1e-12_100000_9_1.0e-02_1.0e+15_l1_noise.pkl",
]

path_gbms = [
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_0_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_1_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_2_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_3_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_4_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_5_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_6_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_7_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_8_1.0e-02_1.0e+15_l1_noise.pkl",
    "/root/meta-pde-solver/src/solvers/raw_records_p35og8zg_1e-12_100000_9_1.0e-02_1.0e+15_l1_noise.pkl",
]
#%%
res_baseline = []
for path in path_baseline:
    res_baseline.append(pickle.load(open(path, "rb")))

#%%
res_gbms = []
for path in path_gbms:
    res_gbms.append(pickle.load(open(path, "rb")))


#%%
for title, res in [("baseline", res_baseline), ("gbms", res_gbms)]:
    i = 0
    total_nits = []
    plt.figure(figsize=(10, 8))
    for ind_hist, fit_hist in res:
        total_nit = 0
        gen = []
        for pop, results in zip(ind_hist, fit_hist):
            nit_hists = torch.stack([fit[1] for fit in results if not len(fit) == 1])
            omega_hists = [fit[2] for fit in results if not len(fit) == 1]
            total_nit += nit_hists.sum()
            gen.append(nit_hists.sum(axis=1).mean())
        plt.plot(gen, alpha=0.5, label=f"seed {i} (#it:{total_nit:.2g}, #gen:{len(gen)})")
        plt.ylim(0, 100000)
        # plt.yscale("log")
        i += 1
        total_nits.append(total_nit)
    plt.legend()
    plt.title(title + f" (mean:{np.mean(total_nits):.3g}, std:{np.std(total_nits):.3g})")
    plt.xlabel("generation")
    plt.ylabel("number of iterations")
    plt.show()
# %%
for title, res in [("baseline", res_baseline), ("gbms", res_gbms)]:
    i = 0
    for ind_hist, fit_hist in res:
        total_nit = 0
        gen = []
        for pop, results in zip(ind_hist, fit_hist):
            nit_hists = torch.stack([fit[1] for fit in results if not len(fit) == 1])
            omega_hists = [fit[2] for fit in results if not len(fit) == 1]
            total_nit += nit_hists.sum()
            gen.append(nit_hists.sum(axis=1).max())
        plt.plot(gen, alpha=0.5, label=f"seed {i} (#it:{total_nit:.2g}, #gen:{len(gen)})")
        plt.ylim(0, 1000000)
        # plt.yscale("log")
        i += 1
    plt.legend()
    plt.title(title)
    plt.show()
# %%
for title, res in [("baseline", res_baseline), ("gbms", res_gbms)]:
    i = 0
    for ind_hist, fit_hist in res:
        total_nit = 0
        gen = []
        for pop, results in zip(ind_hist, fit_hist):
            objectives = torch.stack([ind.fitness.getValues()[0] for ind in pop])
            gen.append(objectives.min())
        plt.plot(gen, alpha=0.5, label=f"seed {i} ( #gen:{len(gen)})")
        plt.ylim(1e-4, 1)
        plt.yscale("log")
        i += 1
    plt.legend()
    plt.title(title)
    plt.show()
#%%
max_iter = 1000
total_nits = []
for title, res in [("baseline", res_baseline), ("gbms", res_gbms)]:
    i = 0
    for ind_hist, fit_hist in res:
        total_nit = 0
        gen = []
        for pop, results in zip(ind_hist, fit_hist):
            nit_hists = torch.stack([fit[1] for fit in results if not len(fit) == 1])
            # nit_hists[nit_hists > max_iter] = max_iter
            omega_hists = [fit[2] for fit in results if not len(fit) == 1]
            total_nit += nit_hists.sum()
            gen.append(nit_hists.sum(axis=1).mean())
        plt.plot(gen, alpha=0.5, label=f"seed {i} (#it:{total_nit:.2g}, #gen:{len(gen)})")
        plt.ylim(0, 100000)
        # plt.yscale("log")
        i += 1
        total_nits.append(total_nit)
    plt.legend()
    plt.title(title + f" (mean:{np.mean(total_nits):.2g}, std:{np.std(total_nits):.2g})")
    plt.show()
# %%
