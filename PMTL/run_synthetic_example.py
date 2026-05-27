import sys

import autograd.numpy as np
import yaml
from autograd import grad
from matplotlib import pyplot as plt

from solvers.min_norm_solvers_numpy import MinNormSolver
from utils.preference import circle_points

# ── Pareto gradient directions ────────────────────────────────────────────────
# Note: these NumPy versions normalize preference vectors to unit norm before
# computing the constraint matrix w. This differs from the PyTorch version in
# train.py, where ref_vec is already unit-norm from circle_points([1], ...).


def get_d_moomtl(grads):
    """Gradient direction for MOO-MTL: min-norm element in gradient convex hull."""
    sol, _ = MinNormSolver.find_min_norm_element(grads)
    return sol


def get_d_paretomtl(grads, value, weights, i):
    """Gradient direction for the main ParetoMTL phase."""
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, i, axis=0) / np.linalg.norm(
        np.delete(weights, i, axis=0), axis=1, keepdims=True
    )
    w = normalized_rest_weights - normalized_current_weight

    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    vec = np.concatenate((grads, np.dot(w[idx], grads)), axis=0)
    sol, _ = MinNormSolver.find_min_norm_element(vec)

    weight0 = sol[0] + np.sum(
        np.array([sol[j] * w[idx][j - 2, 0] for j in np.arange(2, 2 + np.sum(idx))])
    )
    weight1 = sol[1] + np.sum(
        np.array([sol[j] * w[idx][j - 2, 1] for j in np.arange(2, 2 + np.sum(idx))])
    )
    return np.stack([weight0, weight1])


def get_d_paretomtl_init(grads, value, weights, i):
    """Gradient direction for the ParetoMTL initialization phase."""
    nobj, dim = grads.shape
    normalized_current_weight = weights[i] / np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, i, axis=0) / np.linalg.norm(
        np.delete(weights, i, axis=0), axis=1, keepdims=True
    )
    w = normalized_rest_weights - normalized_current_weight

    gx = np.dot(w, value / np.linalg.norm(value))
    idx = gx > 0

    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec = np.dot(w[idx], grads)
        sol, _ = MinNormSolver.find_min_norm_element(vec)

    weight0 = np.sum(
        np.array([sol[j] * w[idx][j, 0] for j in np.arange(0, np.sum(idx))])
    )
    weight1 = np.sum(
        np.array([sol[j] * w[idx][j, 1] for j in np.arange(0, np.sum(idx))])
    )
    return np.stack([weight0, weight1])


# ── Synthetic problem definition ──────────────────────────────────────────────
# Two objectives with opposing optima in R^n. The Pareto front is a concave
# curve, making it a standard benchmark for multi-objective optimization.


def f1(x):
    n = len(x)
    return 1 - np.exp(-np.sum([(x[i] - 1.0 / np.sqrt(n)) ** 2 for i in range(n)]))


def f2(x):
    n = len(x)
    return 1 - np.exp(-np.sum([(x[i] + 1.0 / np.sqrt(n)) ** 2 for i in range(n)]))


f1_dx = grad(f1)
f2_dx = grad(f2)


def concave_fun_eval(x):
    return np.stack([f1(x), f2(x)]), np.stack([f1_dx(x), f2_dx(x)])


def create_pf():
    """Compute the ground-truth Pareto front by sampling the diagonal x1=x2."""
    ps = np.linspace(-1 / np.sqrt(2), 1 / np.sqrt(2))
    pf = []
    for x1 in ps:
        x = np.array([x1, x1])
        f, _ = concave_fun_eval(x)
        pf.append(f)
    return np.array(pf)


# ── Optimization methods ──────────────────────────────────────────────────────


def linear_scalarization_search(t_iter=100, n_dim=20, step_size=1):
    """Baseline: gradient descent with a fixed random scalarization weight."""
    r = np.random.rand(1)
    weights = np.stack([r, 1 - r])
    x = np.random.uniform(-0.5, 0.5, n_dim)
    for t in range(t_iter):
        f, f_dx = concave_fun_eval(x)
        x = x - step_size * np.dot(weights.T, f_dx).flatten()
    return x, f


def moo_mtl_search(t_iter=100, n_dim=20, step_size=1):
    """MOO-MTL: dynamic weighting without preference guidance."""
    x = np.random.uniform(-0.5, 0.5, n_dim)
    for t in range(t_iter):
        f, f_dx = concave_fun_eval(x)
        weights = get_d_moomtl(f_dx)
        x = x - step_size * np.dot(weights.T, f_dx).flatten()
    return x, f


def pareto_mtl_search(ref_vecs, i, t_iter=100, n_dim=20, step_size=1):
    """ParetoMTL: two-phase search guided by a preference vector."""
    x = np.random.uniform(-0.5, 0.5, n_dim)

    # Initialization phase (20% of iterations): find a feasible starting point
    for t in range(int(t_iter * 0.2)):
        f, f_dx = concave_fun_eval(x)
        weights = get_d_paretomtl_init(f_dx, f, ref_vecs, i)
        x = x - step_size * np.dot(weights.T, f_dx).flatten()

    # Main phase (80% of iterations): converge to the Pareto-optimal solution
    for t in range(int(t_iter * 0.8)):
        f, f_dx = concave_fun_eval(x)
        weights = get_d_paretomtl(f_dx, f, ref_vecs, i)
        x = x - step_size * np.dot(weights.T, f_dx).flatten()

    return x, f


# ── Entry point ───────────────────────────────────────────────────────────────


def run(method="ParetoMTL", num=10, save_dir="results"):
    """Run the specified method and plot results against the true Pareto front.

    Args:
        method: 'ParetoMTL', 'MOOMTL', or 'Linear'
        num: number of Pareto-optimal solutions to find
        save_dir: directory to save plot and log
    """
    pf = create_pf()
    f_value_list = []
    weights = circle_points([1], [num])[0]

    for i in range(num):
        if method == "ParetoMTL":
            x, f = pareto_mtl_search(ref_vecs=weights, i=i)
        elif method == "MOOMTL":
            x, f = moo_mtl_search()
        elif method == "Linear":
            x, f = linear_scalarization_search()
        f_value_list.append(f)
        print("Solution {}/{}: f1={:.4f}, f2={:.4f}".format(i + 1, num, f[0], f[1]))

    f_value = np.array(f_value_list)

    import os

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(pf[:, 0], pf[:, 1], label="Pareto front")
    ax.scatter(f_value[:, 0], f_value[:, 1], c="r", s=50, linewidths=0.5, label=method)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("{}".format(method))
    ax.legend()
    fig.savefig(
        os.path.join(save_dir, "{}_num{}.png".format(method, num)),
        dpi=150,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run(**cfg)
