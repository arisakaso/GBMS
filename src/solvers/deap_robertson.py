#%%
from scoop import futures
import multiprocessing
import random
from functools import partial

from models.train_rober import GBMSRober
from src.utils.utils import load_model
import numpy as np
import torch
from deap import base, creator, tools

from src.solvers.newton import Newton, NewtonSOR, NewtonSORJit, compute_F, compute_J, solve_robertson
import pickle


from multiprocessing import set_start_method

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def evalRober(individual, Y_observed, params, save_dir=None):
    """[summary]

    Args:
        individual ([type]): [description]
        y_observed ([type]): [description]
        params ([type]): [description]
        save_dir ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    # print(individual)
    k1, k2, k3 = 10 ** torch.tensor(individual).unsqueeze(-1)
    # print(individual.i, k1, k2, k3)
    # if any(k < 0 for k in individual):
    #     return (torch.tensor(1e5),)
    y_init, hs, tol, max_iter, omega, meta_learner = params
    if save_dir:
        save_dir += f"{individual.g}_{individual.i}"
    with torch.no_grad():
        Y_simulated, nit_hist, omega_hist = solve_robertson(
            k1,
            k2,
            k3,
            y_init,
            omega,
            hs,
            tol,
            max_iter,
            "newton_sor_jit",
            meta_learner=meta_learner,
            save_dir=save_dir,
        )
    # loss = torch.sum(torch.linalg.norm(Y_observed - Y_simulated, axis=2) / torch.linalg.norm(Y_observed, axis=2))
    loss = (1 - (Y_simulated + 1e-14) / (Y_observed + 1e-14)).abs().mean()
    # print(nit_hist.mean())
    return (loss, nit_hist, omega_hist)


def feasible(individual):
    return all(k > 0 for k in individual)


def lognuniform(low=0, high=1):
    return np.power(10, random.uniform(low, high))


def initPopulation(pcls, ind_init, file):
    return pcls(ind_init(c) for c in file)


#%%

# True params
batchsize = 1

k1_, k2_, k3_ = 4e-2, 3e7, 1e4
k1 = torch.ones((batchsize)) * k1_
k2 = torch.ones((batchsize)) * k2_
k3 = torch.ones((batchsize)) * k3_

y_init = torch.zeros((batchsize, 3))
y_init[:, 0] = 1
h0 = 1e-6
n = 100
t = np.geomspace(h0, 1e4, n + 1) - h0
hs = t[1:] - t[:-1]
t = t[1:]
hs = torch.ones(batchsize, n) * hs
omega = torch.ones((batchsize, 1)) * 1.37
tol = 1e-12


#%%
# save_dir = "/root/meta-pde-solver/data_share/raw/deap2/"
# print("load model")
save_dir = None
# run_path = "sohei/rober5/p35og8zg"
# run_path = "sohei/rober5/cmjce6vm"
# gbms = load_model(GBMSRober, run_path, "best").eval()
gbms = None
if gbms:
    meta_learner = gbms.meta_learner
    run_id = run_path.split("/")[-1]
else:
    meta_learner = None
    run_id = "baseline"
with torch.no_grad():
    Y_true, nit_hist, omega_hist = solve_robertson(
        k1,
        k2,
        k3,
        y_init,
        torch.ones((batchsize, 1)),
        hs,
        1e-15,
        1000000000,
        "newton",
        meta_learner=None,
        save_dir=None,
        zero_initial_guess=False,
    )
max_iter = 100000
torch.manual_seed(0)
noise = torch.normal(mean=torch.zeros_like(Y_true), std=Y_true * 0.01)
Y_observed = Y_true + noise

#%%
evalfunc = partial(
    evalRober, Y_observed=Y_observed, params=(y_init, hs, tol, max_iter, omega, meta_learner), save_dir=save_dir
)


#%%
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # ???????????????????????????
creator.create("Individual", list, fitness=creator.FitnessMin)  # ????????????????????????
toolbox = base.Toolbox()  # Toolbox?????????
toolbox.register("attr_gene", random.uniform, -4, 7)  # ??????????????????????????????"attr_gene"?????????
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, 3)  # ??????????????????????????????individual"?????????
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, 3)  # ??????????????????????????????individual"?????????

# toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # ?????????????????????????????????"population"?????????
toolbox.register("evaluate", evalfunc)  # ????????????"evaluate"?????????
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # ?????????????????????"mate"?????????
toolbox.register(
    "mutate", tools.mutGaussian, mu=[0.0, 0.0, 0.0], sigma=[1e-2, 1e-2, 1e-2], indpb=0.5
)  # ?????????????????????"mutate"?????????
toolbox.register("select", tools.selTournament, tournsize=3)  # ???????????????"select"?????????

if __name__ == "__main__":
    for seed in range(10):
        # ?????????
        toolbox.register("map", futures.map)
        ind_hist = []
        fit_hist = []

        # multiprocessing.freeze_support()
        random.seed(seed)
        POP_SIZE = 500
        L00 = []
        for i in range(POP_SIZE):
            k1 = random.uniform(-4, 0)
            k2 = random.uniform(5, 9)
            k3 = random.uniform(2, 6)
            L00.append([k1, k2, k3])

        toolbox.register("population_guess", initPopulation, list, creator.Individual, L00)

        # GA???????????????
        N_GEN = 100000
        POP_SIZE = 500
        CX_PB = 0.5
        MUT_PB = 0.5
        TOL = 1e-2
        BUDGET = 1e15

        print(f"raw_records_{run_id}_{tol}_{max_iter}_{seed}_{TOL:.1e}_{BUDGET:.1e}.pkl")
        # ?????????????????????
        # pop = toolbox.population(n=POP_SIZE)
        pop = toolbox.population_guess()

        print("Start of evolution")
        g = 0
        for i, ind in enumerate(pop):
            ind.g = g
            ind.i = i

        # ?????????????????????????????????
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        ind_tmp = []
        fit_tmp = []
        for ind, fit in zip(pop, fitnesses):
            ind_tmp.append(ind)
            fit_tmp.append(fit)
            ind.fitness.values = (fit[0],)
        ind_hist.append(ind_tmp)
        fit_hist.append(fit_tmp)
        print("  Evaluated %i individuals" % len(pop))

        # ??????????????????
        fits = [ind.fitness.values[0] for ind in pop]
        # %%
        # ?????????????????????
        while g < N_GEN:

            g = g + 1
            for i, ind in enumerate(pop):
                ind.g = g
                ind.i = i
            print("-- Generation %i --" % g)

            # ?????????????????????????????????
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            # ??????
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # ??????????????????????????????
                if random.random() < CX_PB:
                    toolbox.mate(child1, child2)

                    # ????????????????????????????????????????????????
                    del child1.fitness.values
                    del child2.fitness.values

            # ??????
            for mutant in offspring:

                # ??????????????????????????????
                if random.random() < MUT_PB:
                    toolbox.mutate(mutant)

                    # ????????????????????????????????????????????????
                    del mutant.fitness.values

            # ????????????????????????????????????????????????????????????????????????
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

            ind_tmp = []
            fit_tmp = []
            for ind, fit in zip(invalid_ind, fitnesses):
                ind_tmp.append(ind)
                fit_tmp.append(fit)
                ind.fitness.values = (fit[0],)
            ind_hist.append(ind_tmp)
            fit_hist.append(fit_tmp)
            print("  Evaluated %i individuals" % len(invalid_ind))

            # ?????????????????????????????????????????????
            pop[:] = offspring

            # ??????????????????????????????????????????
            fits = [ind.fitness.values[0] for ind in pop]

            # ?????????????????????????????????
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean**2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            if min(fits) < TOL:
                break

        print("-- End of (successful) evolution --")

        # ?????????????????????
        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (10 ** torch.tensor(best_ind).unsqueeze(-1), best_ind.fitness.values))
        # pool.close()
        # %%
        total_nit = 0
        for pop, results in zip(ind_hist, fit_hist):
            nit_hists = [fit[1] for fit in results if not len(fit) == 1]
            omega_hists = [fit[2] for fit in results if not len(fit) == 1]
            total_nit += torch.stack(nit_hists).sum()
        # %%
        print("Total NIT:", total_nit)
        pickle.dump(
            [ind_hist, fit_hist],
            open(f"raw_records_{run_id}_{tol}_{max_iter}_{seed}_{TOL:.1e}_{BUDGET:.1e}_l1_noise.pkl", "wb"),
        )
