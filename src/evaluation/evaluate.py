import matplotlib.pyplot as plt
import wandb
import yaml
from src.data.data_module import PoissonDataModule
from src.data.poisson import get_A
from tqdm import tqdm
import torch
from src.models.train_model import MetaPDESolver
from src.solvers.jacobi import Jacobi

import pickle
import pandas as pd


def evaluate(solver, A, u, f, x0, max_iters):
    with torch.no_grad():
        errors = []
        u_ks = []
        u_k = x0
        for k in tqdm(range(max_iters)):
            errors.append(torch.mean((u - u_k) ** 2))
            u_ks.append(u_k)
            u_k = solver(A=A, b=f.T, x0=u_k.T, num_iter=1).T

    return errors, u_ks


# load data
data_path = "/root/meta-pde-solver/data/processed/poisson_1d_sin1.pkl"
data_name = data_path.split("/")[-1].split(".")[0]
data_module = PoissonDataModule(
    data_path=data_path,
    test_paths=[],
    num_data=10000,
    batch_size=256,
)
for batch in data_module.test_dataloader()[0]:
    f, u = batch
    f = f[:100]
    u = u[:100]

# common
max_iters = 10000
A = get_A(N=512)
single_solver = Jacobi(A=A, tensor_type=torch.FloatTensor)
double_solver = Jacobi(A=A, tensor_type=torch.DoubleTensor)
# u = u.cuda()

# choose runs
df = pd.read_pickle("/root/meta-pde-solver/reports/resut_df.pkl")
temp_df = df.query(
    "train_path == '../../data/processed/poisson_1d_sin1.pkl' \
    and meta_learner == 'fcn_normal' \
    and num_basis == 0"
)
run_paths = list(temp_df.path)
run_paths.append("baseline")
print(run_paths)

for run_path in run_paths:
    if run_path == "baseline":
        x0_fixed = torch.zeros_like(u)
        evaluation_sets = [
            (f"baseline_single_{data_name}", single_solver, A, u, f, x0_fixed, max_iters),
            (f"baseline_double_{data_name}", double_solver, A, u, f, x0_fixed, max_iters),
        ]

    else:
        # get run
        api = wandb.Api()
        run = api.run(run_path)

        # get config and checkpoint
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

        # load model
        meta_pde_solver = MetaPDESolver(conf)
        meta_pde_solver = meta_pde_solver.load_from_checkpoint(ckpt.name)
        meta_pde_solver = meta_pde_solver.eval()
        train_dataset = meta_pde_solver.hparams.train_path.split("/")[-1].split(".")[0]
        model_name = meta_pde_solver.hparams.meta_learner + str(meta_pde_solver.hparams.num_iter) + "_" + train_dataset

        # setup
        x0_nn = meta_pde_solver.meta_learner(f)

        evaluation_sets = [
            (f"{model_name}_single_{data_name}", single_solver, A, u, f, x0_nn, max_iters),
            (f"{model_name}_double_{data_name}", double_solver, A, u, f, x0_nn, max_iters),
        ]

    # evaluate & save
    for name, solver, A, u, f, x0, max_iters in evaluation_sets:
        print(name)
        errors, u_ks = evaluate(solver=solver, A=A, u=u, f=f, x0=x0, max_iters=max_iters)
        pickle.dump(errors, open(f"/root/meta-pde-solver/reports/{name}_error_first100.pkl", "wb"))
        pickle.dump(u_ks, open(f"/root/meta-pde-solver/reports/{name}_uk_first100.pkl", "wb"))
        
