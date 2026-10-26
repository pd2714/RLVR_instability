"""Shared helpers: run a list of simulation configs in parallel and store results."""
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

from rlvr_collapse.sim import SimConfig, run  # noqa: E402

COLLAPSE_PHI = 0.5   # run counts as collapsed if final |mean sign z_i| > 0.5


def _job(cfg):
    t0 = time.time()
    h = run(cfg)
    out = {
        "cfg": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(cfg).items()
                if k != "a" or np.isscalar(v)},
        "final": {k: h[k][-1].tolist() for k in ["phi", "acc", "pass1", "reward", "flipped", "conf", "shift"]},
        "traj": {k: h[k].mean(axis=1).tolist() for k in ["epoch", "phi", "acc", "pass1", "reward", "conf"]
                 if h[k].ndim == 2} | {"epoch": h["epoch"].tolist()},
        "traj_runs": {k: h[k][:, :4].T.tolist() for k in ["phi", "acc", "reward", "shift"]},
        "seconds": time.time() - t0,
    }
    out["p_collapse"] = float(np.mean(np.array(out["final"]["phi"]) > COLLAPSE_PHI))
    return out


def run_sweep(name, cfgs, procs=13):
    # longest jobs first for load balancing
    cost = [c.epochs * c.N * c.runs * (3 if c.opt == "adam" else 1) for c in cfgs]
    order = np.argsort(cost)[::-1]
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(_job, [cfgs[i] for i in order], chunksize=1)
    out = [None] * len(cfgs)
    for i, r in zip(order, res):
        out[i] = r
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(out, f)
    print(f"[{name}] {len(cfgs)} configs in {time.time() - t0:.0f}s")
    return out


def load(name):
    with open(os.path.join(RESULTS, f"{name}.json")) as f:
        return json.load(f)


def fit_Nc(Ns, pc, runs):
    """Fit P(collapse) = sigmoid(k (ln N - ln N_c)) by max likelihood; return N_c."""
    from scipy.optimize import minimize
    x = np.log(np.asarray(Ns, float))
    y = np.asarray(pc, float)
    n = np.asarray(runs, float)

    def nll(th):
        q = 1 / (1 + np.exp(-th[1] * (x - th[0])))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        return -np.sum(n * (y * np.log(q) + (1 - y) * np.log(1 - q)))
    # initialize at the crossing
    x0 = x[np.argmin(np.abs(y - 0.5))]
    best = min((minimize(nll, [x0 + d, 3.0], method="Nelder-Mead") for d in (-0.5, 0, 0.5)), key=lambda r: r.fun)
    return float(np.exp(best.x[0])), float(best.x[1])
