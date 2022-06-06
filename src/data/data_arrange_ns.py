#%%
import pickle
import numpy as np
import pytorch_lightning as pl
from torch._C import dtype
from torch.utils.data import DataLoader
import torch
import os
import glob


def get_geometrymask(N=128):
    G = torch.zeros(N, N)
    G[0, :] = 1
    G[-1, :] = 1
    G[:, 0] = 1
    G[:, -1] = 1
    return G


#%%
# create dataset with previous pressure
ds_paths = glob.glob("../../data/raw/ns_0.01_1e-06*")
print(ds_paths)
for ds_path in ds_paths:

    fs, ps, us, vs, ks, u_tops, u_bottoms = pickle.load(open(ds_path, "rb"))
    f, p = fs[1], ps[1]
    p0 = np.zeros_like(p)
    ps = [p0] + ps
    N = f.shape[0]
    b = torch.zeros_like(torch.tensor(f)).type(torch.float32)
    G = get_geometrymask(N).type(torch.float32)
    ds = [
        (
            torch.stack([torch.tensor(fs[i], dtype=torch.float32), G, b, torch.tensor(ps[i], dtype=torch.float32)]),
            torch.tensor(ps[i + 1], dtype=torch.float32).reshape(1, N, N),
        )
        for i in range(len(fs))
    ]
    save_path = os.path.join("../../data/processed", ("pre_pressure2_" + os.path.basename(ds_path)))
    print(save_path)
    pickle.dump(ds, open(save_path, "wb"))


# %%
