"""Stage A: numerical tests of the collective-collapse theory on the solvable model.

Usage: python experiments/stage_a.py <sweep> [<sweep> ...]
Sweeps: lambda, conf, batch, batch_linear, stable, hetero
"""
import sys

import numpy as np

from common import SimConfig, run_sweep
from rlvr_collapse.theory import amplification_heterogeneous, amplification_per_coupling

LAMBDA_C_GUESS = 3.0  # rough Lambda at the transition (from a pilot run); only sets N grids


def n_grid(N_guess, lo=-4, hi=4, min_N=64):
    Ns = sorted({max(int(round(N_guess * 2 ** (j / 2))), min_N) for j in range(lo, hi + 1)})
    return Ns


def hetero_profiles(N, rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    u = rng.random(N)
    return {
        "unif0-1": 1.0 * u,
        "unif0-2": 2.0 * u,
        "unif0-3": 3.0 * u,
        "unif0.5-2.5": 0.5 + 2.0 * u,
        "mix30": np.where(rng.random(N) < 0.3, 0.3, 2.0),   # 30% uncertain, 70% confident
        "mix60": np.where(rng.random(N) < 0.6, 0.3, 2.0),
        "exp0.7": rng.exponential(0.7, N),
    }


def sweeps(name):
    base = dict(B=32, lr=4.0, epochs=300, runs=24, K=3, adv="reinforce")
    cfgs = []
    if name == "lambda":
        G = amplification_per_coupling(0.5)
        for lam in [0.035, 0.05, 0.07, 0.1, 0.14, 0.2, 0.28]:
            for N in n_grid(LAMBDA_C_GUESS / (lam**2 * G)):
                cfgs.append(SimConfig(N=N, lam=lam, a=0.5, **base))
    elif name == "conf":
        for a in [0.15, 0.3, 0.5, 0.8, 1.1, 1.25]:
            G = amplification_per_coupling(a)
            for N in n_grid(LAMBDA_C_GUESS / (0.1**2 * G)):
                if N <= 32768:
                    cfgs.append(SimConfig(N=N, lam=0.1, a=a, **base))
    elif name == "batch":   # fixed step size; same number of optimizer steps
        for B in [8, 16, 32, 128, 512]:
            for N in [256, 512, 724, 1024, 1448, 2048, 4096, 8192]:
                if N >= 2 * B:
                    cfgs.append(SimConfig(N=N, lam=0.1, a=0.5, **(base | dict(B=B, epochs=300 * B / 32))))
    elif name == "batch_linear":   # linear lr scaling with B; same number of epochs
        for B in [8, 16, 32, 128, 512]:
            for N in [256, 512, 724, 1024, 1448, 2048, 4096, 8192]:
                if N >= 2 * B:
                    cfgs.append(SimConfig(N=N, lam=0.1, a=0.5, **(base | dict(B=B, lr=4.0 * B / 32))))
    elif name == "stable":  # confident initial models; 10x longer training
        for a in [1.4, 1.6, 2.0]:
            for lam in [0.1, 0.3]:
                for N in [1024, 4096]:
                    cfgs.append(SimConfig(N=N, lam=lam, a=a, **(base | dict(epochs=3000, runs=16))))
                cfgs.append(SimConfig(N=16384, lam=lam, a=a, **(base | dict(epochs=600, runs=16))))
        for N in [128, 256]:  # sub-critical uncertain model, trained 10x longer
            cfgs.append(SimConfig(N=N, lam=0.1, a=0.5, **(base | dict(epochs=3000, runs=16))))
    elif name == "hetero":
        for prof in hetero_profiles(8).keys():
            G = amplification_heterogeneous(hetero_profiles(4096)[prof])
            for N in n_grid(LAMBDA_C_GUESS / (0.1**2 * max(G, 0.05)), lo=-4, hi=4):
                N = min(N, 32768)
                c = SimConfig(N=N, lam=0.1, a=hetero_profiles(N)[prof], **base)
                c.extra = {"profile": prof, "G_pop": G}
                cfgs.append(c)
    elif name == "feature":  # feature coupling mu makes local sharpening collective
        for mu in [0.0, 0.02, 0.03, 0.04, 0.05, 0.07]:
            for N in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
                cfgs.append(SimConfig(N=N, lam=0.1, mu=mu, a=0.5, **(base | dict(runs=16))))
    elif name == "remedy":  # marginal-entropy bonus; theory: stable for beta > beta_c(a) <= 0.5
        for beta in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            for N in [512, 2048, 8192]:
                for a in [0.2, 0.5]:
                    cfgs.append(SimConfig(N=N, lam=0.1, beta=beta, a=a, **(base | dict(runs=16))))
    else:
        raise ValueError(name)
    return cfgs


if __name__ == "__main__":
    for name in sys.argv[1:]:
        cfgs = sweeps(name)
        print(name, len(cfgs), "configs; max N", max(c.N for c in cfgs))
        run_sweep(f"A_{name}", cfgs)
