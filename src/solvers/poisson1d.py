#%%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.linalg import solve
from scipy.sparse import diags
from src.data.poisson import get_A
from src.solvers.jacobi1d import Jacobi1d
from sympy import Eq, Function, IndexedBase, exp, lambdify, pi, sin, symbols, tanh


def get_A(N=128, bc_type="d"):
    """create A matrix corresponding to poisson equation

    Args:
        bc_type (str, optional): [description]. Defaults to "d".
        N (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """

    if bc_type == "d":
        A = torch.Tensor(diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray())
        # A *= (N - 1) ** 2
        A[0, 0] = 1  # bc
        A[0, 1] = 0  # bc
        A[-1, -1] = 1  # bc
        A[-1, -2] = 0  # bc
    else:
        A = None

    return A


def get_analytical_f_and_u_sin(num_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    d = IndexedBase("d")
    d2 = IndexedBase("d2")
    e = IndexedBase("e")
    f = IndexedBase("f")
    g = IndexedBase("g")
    h = IndexedBase("h")

    u = 0
    for i in range(num_terms):
        u += a[i] * (
            (x - b[i]) ** c[i] * sin(d[i] * pi * (x - d2[i])) * exp(-((x - e[i] / f[i]) ** 2)) + g[i] * x + h[i]
        )
    ff = u.diff(x, x)

    return ff, u


def get_analytical_f_and_u_sin_simple(num_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    p = IndexedBase("p")
    q = IndexedBase("q")

    u = 0
    for i in range(num_terms):
        u += a[i] * (sin(p[i] * pi * (x - q[i])))
    ff = u.diff(x, x)

    return ff, u


def generate_random_coeffs_sin_simple(num_terms=10, boundary=False):
    x = symbols("x")
    a = IndexedBase("a")
    p = IndexedBase("p")
    q = IndexedBase("q")

    # ais = np.random.randn(num_terms)
    ais = np.random.randn(num_terms)
    pis = np.random.uniform(0, 64, size=num_terms)
    qis = np.random.uniform(0, 1, size=num_terms)

    subs = {
        **{a[i]: ais[i] for i in range(num_terms)},
        **{p[i]: pis[i] for i in range(num_terms)},
        **{q[i]: qis[i] for i in range(num_terms)},
    }

    return subs


# %%
def generate_random_coeffs_sin(num_terms=10, boundary=False):
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    d = IndexedBase("d")
    d2 = IndexedBase("d2")
    e = IndexedBase("e")
    f = IndexedBase("f")
    g = IndexedBase("g")
    h = IndexedBase("h")

    # ais = np.random.randn(num_terms)
    ais = np.random.uniform(-1, 1, size=num_terms)
    bis = np.random.uniform(0, 1, size=num_terms)
    cis = np.random.randint(0, 30, size=num_terms)
    if boundary:
        dis = np.random.uniform(1, 128, size=num_terms)
        d2is = np.random.uniform(0, 1, size=num_terms)
        gis = np.random.uniform(-1, 1, size=num_terms)
        his = np.random.uniform(-1, 1, size=num_terms)
    else:
        dis = np.random.randint(1, 128, size=num_terms)
        d2is = np.zeros(num_terms)
        gis = np.zeros(num_terms)
        his = np.zeros(num_terms)
    eis = np.random.uniform(-0.5, 1.5, size=num_terms)
    fis = np.random.uniform(0, 1, size=num_terms)

    subs = {
        **{a[i]: ais[i] for i in range(num_terms)},
        **{b[i]: bis[i] for i in range(num_terms)},
        **{c[i]: cis[i] for i in range(num_terms)},
        **{d[i]: dis[i] for i in range(num_terms)},
        **{d2[i]: d2is[i] for i in range(num_terms)},
        **{e[i]: eis[i] for i in range(num_terms)},
        **{f[i]: fis[i] for i in range(num_terms)},
        **{g[i]: gis[i] for i in range(num_terms)},
        **{h[i]: his[i] for i in range(num_terms)},
    }

    return subs


# %%
def discretize(expr, N=128):
    """discretize sympy expression in [0, 1]

    Args:
        expr ([type]): [description]
        N (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """
    expr = lambdify(symbols("x"), expr, "numpy")
    return expr(np.linspace(0, 1, N, dtype=np.float64))


# %%
def get_normalized_subs(u, subs, func, num_terms=10, N=512):
    if func == "tanh":
        return subs
    a = IndexedBase("a")
    u = u.xreplace(subs)
    u = discretize(u, N)
    normalizer = np.linalg.norm(u)
    for i in range(num_terms):
        subs[a[i]] /= normalizer
    return subs


# %%
def generate_func(f, u, func, num_terms=10, N=512, computed=False, boundary=False, normalize=False):
    torch.set_default_dtype(torch.float64)
    if func == "sin":
        subs = generate_random_coeffs_sin(num_terms=num_terms, boundary=boundary)
    elif func == "tanh":
        subs = generate_random_coeffs_tanh(num_terms=num_terms, boundary=boundary)
    elif func == "gaussian":
        subs = generate_random_coeffs_gaussian(num_terms=num_terms, boundary=boundary)
    elif func == "sin_simple":
        subs = generate_random_coeffs_sin_simple(num_terms=num_terms, boundary=boundary)
    if normalize:
        subs = get_normalized_subs(u, subs, func, num_terms=num_terms, N=N)
    f = f.xreplace(subs)
    f = discretize(f, N)
    f = torch.from_numpy(f)
    u = u.xreplace(subs)
    u = discretize(u, N)
    u = torch.from_numpy(u)
    if computed:
        A = get_A(N=N)
        f_ = f.clone() / (N - 1) ** 2
        f_[0] = u[0]
        f_[-1] = u[-1]
        u = A.inverse() @ f_
    return f, u, subs


def get_analytical_f_and_u_gaussian(num_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")

    u = 0
    for i in range(num_terms):
        u += a[i] * (exp(-(((x - b[i]) / c[i]) ** 2)))
    ff = u.diff(x, x)

    return ff, u


def generate_random_coeffs_gaussian(num_terms=10, boundary=False):
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")

    ais = np.random.randn(num_terms)
    bis = np.random.uniform(0, 1, size=num_terms)
    cis = np.random.uniform(1e-2, 1, size=num_terms)

    subs = {
        **{a[i]: ais[i] for i in range(num_terms)},
        **{b[i]: bis[i] for i in range(num_terms)},
        **{c[i]: cis[i] for i in range(num_terms)},
    }

    return subs


def get_analytical_f_and_u_tanh(num_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    # b = symbols("b")
    # c = symbols("c")
    # d = symbols("d")
    # e = symbols("e")
    # f = symbols("f")
    p = IndexedBase("p")
    q = IndexedBase("q")
    # r = IndexedBase("r")

    u = 1
    for i in range(num_terms):
        u += a[i] * tanh(p[i] * pi * (x - q[i]))
    # u += b * (tanh(c * pi * x) * tanh(d * pi * (x - 1)) * tanh(e * pi * (x - f)))  # *tanh(g * pi * (x-h)))
    ff = u.diff(x, x)

    return ff, u


def generate_random_coeffs_tanh(num_terms=10, boundary=False):
    a = IndexedBase("a")
    # b = symbols("b")
    # c = symbols("c")
    # d = symbols("d")
    # e = symbols("e")
    # f = symbols("f")
    # g = symbols("g")
    # h = symbols("h")
    p = IndexedBase("p")
    q = IndexedBase("q")

    ais = np.random.randn(num_terms)
    # b_ = np.random.randn()
    # c_ = np.random.uniform(0, 20)
    # d_ = np.random.uniform(0, 20)
    # e_ = np.random.uniform(0, 20)
    # f_ = np.random.uniform(-0.25, 1.25)
    #     g_ = np.random.uniform(0,100)
    #     h_ = np.random.uniform(-0.25, 1.25)
    pis = np.random.uniform(0, 30, size=num_terms)
    qis = np.random.uniform(0, 1, size=num_terms)
    # qis[0] = 0
    # qis[-1] = 1

    subs = {
        **{a[i]: ais[i] for i in range(num_terms)},
        # **{b: b_},
        # **{c: c_},
        # **{d: d_},
        # **{e: e_},
        # **{f: f_},
        #         **{g: g_},
        #         **{h: h_},
        **{p[i]: pis[i] for i in range(num_terms)},
        **{q[i]: qis[i] for i in range(num_terms)},
    }
    return subs


# %%
### CHECK ###


def check_func(f, u):
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

    plt.plot(u, alpha=0.5, label="u")
    plt.plot(u_exact, alpha=0.5, label="u_exact")
    plt.plot(u_hat.reshape(-1), alpha=0.5, label="u_jacobi")
    plt.legend()
    plt.show()

    f_norm = torch.norm(f)
    f = f / f_norm
    dbc = u.reshape(1, 1, -1) / f_norm
    u0 = torch.linspace(u[0], u[-1], N).reshape(1, 1, -1) / f_norm
    jacobi = Jacobi1d(h=1 / (N - 1))
    u_hat = jacobi(f, dbc, 500, u0)

    plt.plot(u_hat.reshape(-1), alpha=0.5, label="u_jacobi")
    plt.legend()
    plt.show()

    plt.plot(u, alpha=0.5, label="u")
    plt.plot(u_exact, alpha=0.5, label="u_exact")
    plt.plot(u_hat.reshape(-1) * f_norm, alpha=0.5, label="u_jacobi")
    plt.legend()
    plt.show()


#%%
N = 512

num_terms = 20
f_a, u_a = get_analytical_f_and_u_tanh(num_terms=num_terms)
# %%
f, u, subs = generate_func(f_a, u_a, "tanh", num_terms=num_terms, N=N, computed=True, boundary=False)
check_func(f, u)
# %%
# num_terms = 20
# f_a, u_a = get_analytical_f_and_u_sin(num_terms=num_terms)
# # %%
# f, u, subs = generate_func(f_a, u_a, "sin", num_terms=num_terms, N=N, computed=True, boundary=False)
# check_func(f, u)
#%%
num_terms = 20
f_a, u_a = get_analytical_f_and_u_sin_simple(num_terms=num_terms)
# %%
f, u, subs = generate_func(f_a, u_a, "sin_simple", num_terms=num_terms, N=N, computed=True, boundary=False)
check_func(f, u)

# %%
num_terms = 20
f_a, u_a = get_analytical_f_and_u_gaussian(num_terms=num_terms)
#%%
f, u, subs = generate_func(f_a, u_a, "gaussian", num_terms=num_terms, N=N, computed=True, boundary=False)
check_func(f, u)
# %%

### GENERATE DATA ###
N = 512
num_date = 50000
funcs_and_num_terms = [
    # ["gaussian", 20],
    ["sin_simple", 20],
    # ["tanh", 20],
]
# funcs_and_num_terms = [["sin_simple", 20]]


for func, num_terms in funcs_and_num_terms:
    if func == "sin":
        f_a, u_a = get_analytical_f_and_u_sin(num_terms=num_terms)
    elif func == "tanh":
        f_a, u_a = get_analytical_f_and_u_tanh(num_terms=num_terms)
    elif func == "gaussian":
        f_a, u_a = get_analytical_f_and_u_gaussian(num_terms=num_terms)
    elif func == "sin_simple":
        f_a, u_a = get_analytical_f_and_u_sin_simple(num_terms=num_terms)
    ds = Parallel(verbose=10, n_jobs=32)(
        [
            delayed(generate_func)(f_a, u_a, func, N=N, num_terms=num_terms, computed=True, boundary=False)
            for i in range(num_date)
        ]
    )

    # SAVE
    meta_df = pd.DataFrame([d[2] for d in ds])
    ds = [[d[0], d[1]] for d in ds]
    save_dir = "/root/meta-pde-solver/data_share/raw/poisson1d"

    ds_path = os.path.join(save_dir, f"{func}_{num_terms}_20210924.pkl")
    meta_df_path = os.path.join(save_dir, f"{func}_{num_terms}_20210924_metadf.pkl")
    pickle.dump(ds, open(ds_path, "wb"))
    pickle.dump(meta_df, open(meta_df_path, "wb"))

# %%
