"""Stage B: robustness of the mechanism to GRPO normalization, group size K,
multiple candidate answers M, and the Adam optimizer.

Usage: python experiments/stage_b.py <sweep> [...]   sweeps: grpo, multi, adam
"""
import sys

import numpy as np

from common import SimConfig, run_sweep
from rlvr_collapse import multiclass
from rlvr_collapse.theory import amplification_per_coupling
from stage_a import LAMBDA_C_GUESS, n_grid


def sweeps(name):
    base = dict(B=32, lr=4.0, epochs=300, runs=24, a=0.5, lam=0.1)
    cfgs = []
    if name == "grpo":
        # step sizes rescaled so the peak drift F* is comparable across estimators
        for adv, K, lr in [("grpo", 3, 4.0 / np.sqrt(2)), ("grpo", 5, 2.0), ("grpo", 8, 1.5),
                           ("mean", 8, 3.0), ("reinforce", 8, 2.5), ("grpo", 16, 1.0)]:
            G = amplification_per_coupling(0.5, K, adv)
            for N in n_grid(LAMBDA_C_GUESS / (0.1**2 * G)):
                c = SimConfig(N=N, K=K, adv=adv, **(base | dict(lr=lr)))
                c.extra = {"G": float(G)}
                cfgs.append(c)
    elif name == "multi":
        for M, K in [(3, 3), (4, 5), (6, 5)]:
            G = multiclass.amplification(0.5, M, K)
            for N in n_grid(LAMBDA_C_GUESS / (0.1**2 * G)):
                c = SimConfig(N=N, M=M, K=K, **base)
                c.extra = {"G": float(G)}
                cfgs.append(c)
    elif name == "adam":
        for lam in [0.2, 0.3, 0.5, 0.7, 1.0]:
            for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
                ep = max(150.0, 150.0 * np.sqrt(1024 / N))
                cfgs.append(SimConfig(N=N, opt="adam", **(base | dict(lam=lam, lr=0.01, epochs=ep, runs=16))))
    else:
        raise ValueError(name)
    return cfgs


if __name__ == "__main__":
    for name in sys.argv[1:]:
        cfgs = sweeps(name)
        print(name, len(cfgs), "configs; max N", max(c.N for c in cfgs), flush=True)
        run_sweep(f"B_{name}", cfgs)
