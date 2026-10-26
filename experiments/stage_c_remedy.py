"""Stage C remedy: marginal-entropy bonus on a Transformer that otherwise collapses."""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import torch

import stage_c as C
from common import RESULTS
from rlvr_collapse.nn import majority_vote_rl

COND = ("plain", 90, 2.0)
BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
NS = [64, 512]
SEEDS = range(4)


def _job(args):
    beta, N, seed, k_d = args
    torch.set_num_threads(1)
    xp, yp, xt, yt = C.data()
    arch, base, tau = COND
    m = C.load_model(arch, base)
    t0 = time.time()
    h = majority_vote_rl(m, xp[:N], xt, yp[:N], yt, tau=tau, K=3, B=C.B_RL, lr=0.125 * C.B_RL / k_d,
                         epochs=C.EPOCHS, seed=seed, n_log=50, beta=beta)
    return dict(beta=beta, N=N, seed=seed, seconds=time.time() - t0,
                final={k: float(v[-1]) for k, v in h.items()},
                traj={k: np.round(v, 4).tolist() for k, v in h.items()})


if __name__ == "__main__":
    kern = json.load(open(os.path.join(RESULTS, "C_kernel.json")))
    k_d = [r for r in kern if (r["arch"], r["base"], r["tau"], r["rho"]) == COND + (0.0,)][0]["k_d"]
    jobs = [(b, N, s, k_d) for b in BETAS for N in NS for s in SEEDS]
    with Pool(13) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(RESULTS, "C_remedy.json"), "w"))
    for b in BETAS:
        for N in NS:
            rs = [r for r in res if r["beta"] == b and r["N"] == N]
            print(b, N, "P(collapse)", np.mean([r["final"]["phi"] > 0.5 for r in rs]),
                  "acc", np.round(np.mean([r["final"]["acc"] for r in rs]), 3),
                  "test", np.round(np.mean([r["final"]["test_acc"] for r in rs]), 3), flush=True)
