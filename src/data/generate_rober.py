#%%
import numpy as np
import torch
import random
import joblib
from src.solvers.newton import solve_robertson
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_num_threads(1)


def lognuniform(low=0, high=1):
    return np.power(10, np.random.uniform(low, high, 3))


torch.set_default_dtype(torch.float64)

# %%
num_samples = 10000
# ks = torch.tensor(np.power(10, np.random.uniform(-8, 8, (num_samples, 3))))
k1s = torch.tensor(np.power(10, np.random.uniform(-4, 0, (num_samples))))
k2s = torch.tensor(np.power(10, np.random.uniform(5, 9, (num_samples))))
k3s = torch.tensor(np.power(10, np.random.uniform(2, 6, (num_samples))))
# %%
batchsize = 1
y_init = torch.zeros((batchsize, 3))
y_init[:, 0] = 1
h0 = 1e-6
n = 100
t = np.geomspace(h0, 1e4, n + 1) - h0
hs = t[1:] - t[:-1]
t = t[1:]
hs = torch.ones(batchsize, n) * hs
omega = torch.ones((batchsize, 1)) * 1
tol = 1e-12
max_iter = 1000000

k1_, k2_, k3_ = 4e-2, 3e7, 1e4
k1 = torch.ones((batchsize)) * k1_
k2 = torch.ones((batchsize)) * k2_
k3 = torch.ones((batchsize)) * k3_
omega = torch.ones((batchsize, 1)) * 1.37

# %%
Y, nit_hist, omega_hist = solve_robertson(
    k1, k2, k3, y_init, omega, hs, tol, max_iter, "newton_sor_jit", meta_learner=None, save_dir=None
)
plt.scatter(t, Y[0, 0, 1:], s=1)
plt.scatter(t, Y[0, 1, 1:], s=1)
plt.scatter(t, Y[0, 2, 1:], s=1)
plt.yscale("log")
plt.xscale("log")
plt.show()
plt.plot(nit_hist)
plt.ylim(1, 30)
plt.show()
#%%
save_dir = "/root/meta-pde-solver/data/raw/rober_sor/"


result = joblib.Parallel(n_jobs=-1, verbose=5)(
    joblib.delayed(solve_robertson)(
        k1s[[j]],
        k2s[[j]],
        k3s[[j]],
        y_init,
        omega,
        hs,
        tol,
        max_iter,
        "newton_sor_jit",
        meta_learner=None,
        save_dir=save_dir + f"{j}_",
    )
    for j in range(num_samples)
)

# %%
# %%
