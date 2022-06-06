import numpy as np
import torch
from scipy.sparse import diags
from sympy import symbols, Function, IndexedBase, sin, pi, exp, Eq, lambdify


def generate_poisson_eq(num_terms: int = 10):
    """generate poisson equation with N sin functions


    Args:
        N (int, optional): [description]. Defaults to 10.

    Returns:
        poisson equation

    Examples:
        >>> generate_poisson_eq(num_terms=3)
        Eq(Derivative(u(x), (x, 2)), sin(pi*x*b[0])*a[0] + sin(pi*x*b[1])*a[1] + sin(pi*x*b[2])*a[2])

    """
    x = symbols("x")
    u = symbols("u", cls=Function)
    a = IndexedBase("a")
    b = IndexedBase("b")
    f = 0
    for i in range(num_terms):
        f += a[i] * sin(b[i] * pi * x)

    return Eq(u(x).diff(x, x), f)


def get_analytical_f(num_terms: int = 10):
    """get analytical f

    Args:
        N (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]

    Example:
        >>> get_analytical_f(num_terms=2)
        sin(pi*x*b[0])*a[0] + sin(pi*x*b[1])*a[1]
    """
    f = 0
    a = IndexedBase("a")
    b = IndexedBase("b")
    x = symbols("x")

    for i in range(num_terms):
        f += a[i] * sin(b[i] * pi * x)

    return f


def get_analytical_u(num_terms: int = 10):
    """get analytical solution

    Args:
        N (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: analytical solution with length N

    Example:
        >>> get_analytical_u(num_terms=2)
        -sin(pi*x*b[0])*a[0]/(pi**2*b[0]**2) - sin(pi*x*b[1])*a[1]/(pi**2*b[1]**2)
    """
    u = 0
    a = IndexedBase("a")
    b = IndexedBase("b")
    x = symbols("x")

    for i in range(num_terms):
        u -= sin(b[i] * pi * x) * a[i] / ((pi ** 2) * (b[i] ** 2))

    return u


def get_analytical_f_and_u(num_terms: int = 10):
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
    e = IndexedBase("e")
    f = IndexedBase("f")
    g = IndexedBase("g")

    u = 0
    for i in range(num_terms):
        u += a[i] * (x - b[i]) ** c[i] * sin(d[i] * pi * x) * exp(-((x - e[i] / f[i]) ** 2))
    # u = u ** 3
    ff = u.diff(x, x)

    return ff, u


def assign_random_coeffs_2(ff, u, num_terms=10):
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")
    d = IndexedBase("d")
    e = IndexedBase("e")
    f = IndexedBase("f")

    ais = np.random.randn(num_terms)
    bis = np.random.uniform(-0.12, 1.12, size=num_terms)
    cis = np.random.randint(0, 30, size=num_terms)
    dis = np.random.randint(1, 100, size=num_terms)
    eis = np.random.uniform(0, 1, size=num_terms)
    fis = np.random.uniform(0, 1, size=num_terms)

    subs = {
        **{a[i]: ais[i] for i in range(num_terms)},
        **{b[i]: bis[i] for i in range(num_terms)},
        **{c[i]: cis[i] for i in range(num_terms)},
        **{d[i]: dis[i] for i in range(num_terms)},
        **{e[i]: eis[i] for i in range(num_terms)},
        **{f[i]: fis[i] for i in range(num_terms)},
    }

    ff = ff.xreplace(subs)
    u = u.xreplace(subs)

    return ff, u


def assign_random_coeffs(f, u, p_range=(1, 100), c_range=(-10, 10)):
    """assign random coefficients to f and u

    Args:
        f ([type]): [description]
        u ([type]): [description]

    Returns:
        assigned f and u

    """

    a = IndexedBase("a")
    b = IndexedBase("b")
    N = len(f.args) + 1
    periods = np.random.randint(low=p_range[0], high=p_range[1], size=N)
    coeffs = np.random.uniform(low=c_range[0], high=c_range[1], size=N)
    subs = {
        **{a[i]: coeffs[i] for i in range(N)},
        **{b[i]: periods[i] for i in range(N)},
    }
    f = f.subs(subs)
    u = u.subs(subs)

    return f, u


def discretize(expr, N=128):
    """discretize sympy expression in [0, 1]

    Args:
        expr ([type]): [description]
        N (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """
    expr = lambdify(symbols("x"), expr, "numpy")
    return expr(np.linspace(0, 1, N))


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
        A[0, 0] = 1  # bc
        A[0, 1] = 0  # bc
        A[-1, -1] = 1  # bc
        A[-1, -2] = 0  # bc
    else:
        A = None

    return A


def discretize_1d_poisson_eauation(f=0, bc_type="d", bc=(0, 0), N=128):

    A = get_A(bc_type, N)
    f = discretize(f, N) / (N ** 2)
    f[[0, -1]] = bc
    f = torch.Tensor(f)

    return A, f


def generate_data(f, u, p_range=(1, 100), c_range=(-10, 10), bc=(0, 0), N=128):
    f, u = assign_random_coeffs(f, u, p_range, c_range)
    u = discretize(u, N)
    f = discretize(f, N)  # / (N**2)
    f[0] = bc[0]
    f[-1] = bc[-1]

    return torch.Tensor(f), torch.Tensor(u)


def generate_data_2(f, u, num_terms=20, N=128, bc=(0, 0)):
    f, u = assign_random_coeffs_2(f, u, num_terms)
    u = discretize(u, N)
    f = discretize(f, N)
    f[0] = bc[0]
    f[-1] = bc[-1]

    return torch.Tensor(f), torch.Tensor(u)
