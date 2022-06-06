#%%
import os
import pickle

import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.linalg import solve
from src.solvers.jacobi1d import Jacobi1d
from src.solvers.poisson1d import (generate_normalized_func, get_A,
                                   get_analytical_f_and_u_gaussian,
                                   get_analytical_f_and_u_sin)

# %%
torch.set_default_dtype(torch.float64)

N = 512
boundary = False
f_a, u_a = get_analytical_f_and_u_sin(num_terms=100)

#%%
# CHECK
f, u, subs = generate_normalized_func(f_a, u_a, num_terms=100, N=N, computed=True, boundary=False)
A = get_A(N=N)
f_ = f.clone() / (N - 1) ** 2
f_[0] = u[0]
f_[-1] = u[-1]
u_exact = A.inverse() @ f_
# u_scipy = solve(A.numpy(), f_.numpy())
plt.plot(f, alpha=0.5)
plt.show()

f = f.reshape(1, 1, -1)
dbc = u.reshape(1, 1, -1)
u0 = torch.linspace(u[0], u[-1], N).reshape(1, 1, -1)
jacobi = Jacobi1d(h=1 / (N - 1))
u_hat = jacobi(f, dbc, 500, u0)


plt.plot(u, alpha=0.5)
plt.plot(u_exact, alpha=0.5)
plt.plot(u_hat.reshape(-1), alpha=0.5)
plt.ylim(-0.2, 0.2)
plt.show()
# %%
# GENERATE DATA
num_terms = 100
N = 512
num_date = 30000
boundaries = [False, True]
computed = True

f_a, u_a = get_analytical_f_and_u_sin(num_terms=num_terms)

for boundary in boundaries:
    ds = Parallel(verbose=10, n_jobs=8)(
        [
            delayed(generate_normalized_func)(f_a, u_a, N=N, num_terms=num_terms, computed=computed, boundary=boundary)
            for i in range(num_date)
        ]
    )

    # SAVE
    meta_df = pd.DataFrame([d[2] for d in ds])
    ds = [[d[0], d[1]] for d in ds]
    save_dir = "/root/meta-pde-solver/data_share/raw/poisson1d_sin"

    ds_path = os.path.join(save_dir, f"{num_terms}_{computed}_{boundary}_20210628.pkl")
    meta_df_path = os.path.join(save_dir, f"{num_terms}_{computed}_{boundary}_20210628_metadf.pkl")
    pickle.dump(ds, open(ds_path, "wb"))
    pickle.dump(meta_df, open(meta_df_path, "wb"))
# %%
num_terms = 1
N = 512
num_date = 30000
boundaries = [False, True]
computed = True

f_a, u_a = get_analytical_f_and_u_gaussian(num_terms=num_terms)
# %%
f, u, subs = generate_normalized_func(f_a, u_a, "gaussian", num_terms=num_terms, N=N, computed=False, boundary=False)
plt.plot(f, alpha=0.5)
plt.show()
plt.plot(u, alpha=0.5)
# %%
subs
# %%
