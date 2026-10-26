"""Stage C: does the kernel-measured instability predict collapse in a real network?

Two architectures (see rlvr_collapse/nn.py):
  plain   : CLS readout; strong intrinsic shared coupling k_u.
  antisym : z(x) = f(x) - f(swap x); intrinsic k_u ~ 0.  A controlled shared
            coupling is re-introduced through a scaled output bias, rho = lam_b^2 / k_d.
For each condition (arch, base checkpoint, temperature tau, rho) we measure the
empirical-NTK statistics k_d, k_u, k_s and the initial logits on the unlabeled
pool, compute the predicted amplification Lambda(N) = G N k_u / (k_d' + N k_s),
and run label-free majority-vote RL for a grid of N and seeds.

Usage: python experiments/stage_c.py [bases|measure|run|all]
"""
import copy
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

from common import RESULTS
from rlvr_collapse.nn import TinyTransformer, kernel_stats, majority_vote_rl, make_data, pretrain
from rlvr_collapse.theory import amplification_heterogeneous, instability_index

CKPT = os.path.join(RESULTS, "nn_bases.pt")
B_RL = 16
EPOCHS = 100
SEEDS = list(range(6))
CONDITIONS = (
    [("plain", base, tau, 0.0) for base in (90, 120) for tau in (1.0, 2.0, 3.0)]
    + [("antisym", base, tau, rho) for base in (10, 20) for tau in (1.0, 2.0) for rho in (0.0, 0.3, 1.0, 3.0)]
)
NS = {"plain": [16, 32, 64, 128, 256, 512, 1024], "antisym": [32, 64, 128, 256, 512, 1024, 2048]}


def data():
    xp, yp = make_data(8192, np.random.default_rng(7))
    xt, yt = make_data(2000, np.random.default_rng(1))
    return xp, yp, xt, yt


def build_bases():
    _, _, xt, yt = data()
    ck, info = {}, {}
    for arch, steps in [("plain", (90, 120)), ("antisym", (10, 20))]:
        torch.manual_seed(0)
        m = TinyTransformer(antisym=arch == "antisym")
        for t in range(1, max(steps) // 10 + 1):
            pretrain(m, 10, seed=100 + t, lr=1e-3)
            if t * 10 in steps:
                with torch.no_grad():
                    z = m(xt)
                ck[(arch, t * 10)] = copy.deepcopy(m.state_dict())
                info[f"{arch}{t * 10}"] = dict(acc=float(((z > 0).long() == yt).float().mean()),
                                               med_abs_z=float(z.abs().median()))
    torch.save(ck, CKPT)
    json.dump(info, open(os.path.join(RESULTS, "C_bases.json"), "w"), indent=1)
    print(info, flush=True)


def load_model(arch, base, lam_b=0.0):
    m = TinyTransformer(lam_b=lam_b, antisym=arch == "antisym")
    m.load_state_dict(torch.load(CKPT)[(arch, base)])
    return m


def measure():
    torch.set_num_threads(8)
    xp, yp, xt, yt = data()
    out = []
    for arch, base, tau, rho in CONDITIONS:
        kd1 = kernel_stats(load_model(arch, base), xp[:384], 1.0)["k_d"]
        lam_b = float(np.sqrt(rho * kd1))
        m = load_model(arch, base, lam_b)
        ks = kernel_stats(m, xp[:384], tau)
        ks.pop("z")
        with torch.no_grad():
            z = (m(xp[:2048]) / tau).numpy()
        acc0 = float(((torch.tensor(z) > 0).long() == yp[:2048]).float().mean())
        kdp = ks["k_d"] - ks["k_u"] - ks["k_s"]
        G = amplification_heterogeneous(z)
        row = dict(arch=arch, base=base, tau=tau, rho=rho, lam_b=lam_b, **ks, k_dp=kdp, G=G,
                   index=instability_index(z), acc0=acc0, med_abs_z=float(np.median(np.abs(z))),
                   phi0=float(abs(np.sign(z).mean())))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        out.append(row)
    json.dump(out, open(os.path.join(RESULTS, "C_kernel.json"), "w"), indent=1)
    return out


def _job(args):
    arch, base, tau, rho, lam_b, k_d, N, seed = args
    torch.set_num_threads(1)
    xp, yp, xt, yt = data()
    m = load_model(arch, base, lam_b)
    B = min(B_RL, N)
    lr = 0.125 * B / k_d          # per-visit self-kernel step of 0.125, as in the toy model
    t0 = time.time()
    h = majority_vote_rl(m, xp[:N], xt, yp[:N], yt, tau=tau, K=3, B=B, lr=lr, epochs=EPOCHS, seed=seed, n_log=50)
    return dict(arch=arch, base=base, tau=tau, rho=rho, N=N, seed=seed, lr=lr, seconds=time.time() - t0,
                final={k: float(v[-1]) for k, v in h.items()},
                traj={k: np.round(v, 4).tolist() for k, v in h.items()})


def run_all():
    kern = json.load(open(os.path.join(RESULTS, "C_kernel.json")))
    jobs = [(r["arch"], r["base"], r["tau"], r["rho"], r["lam_b"], r["k_d"], N, s)
            for r in kern for N in NS[r["arch"]] for s in SEEDS]
    jobs.sort(key=lambda j: -j[6])
    t0 = time.time()
    res = []
    with Pool(13) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            res.append(r)
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(RESULTS, "C_runs.json"), "w"))
    print(f"{len(jobs)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("bases", "all"):
        build_bases()
    if what in ("measure", "all"):
        measure()
    if what in ("run", "all"):
        run_all()
