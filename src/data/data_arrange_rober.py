#%%
import torch
import pickle
import os
import joblib

#%%
root_dir = "/root/meta-pde-solver/data_tmp/raw/deap2/deap2"
target_dir = "/root/meta-pde-solver/data_tmp/processed/rober"
#%%
def arrange_data(root_dir, target_dir, file):
    y, y_old, h, k1, k2, k3, tol = pickle.load(open(os.path.join(root_dir, file), "rb"))
    x = torch.cat([y_old.squeeze(), torch.Tensor([h, k1, k2, k3])])
    y = y.squeeze()
    x = x.double()
    y = y.double()
    pickle.dump((x, y), open(os.path.join(target_dir, file), "wb"))


res = joblib.Parallel(n_jobs=-1, verbose=1)(
    joblib.delayed(arrange_data)(root_dir, target_dir, file) for file in os.listdir(root_dir)
)
