"""Kernel drift during label-free RL: do k_u/k_d' and k_s/k_d' change before collapse?"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

import stage_c as C
from common import RESULTS
from rlvr_collapse.nn import kernel_stats, majority_vote_rl

CASES = [("antisym", 10, 1.0, 0.3, 1024), ("antisym", 20, 1.0, 0.3, 2048), ("antisym", 20, 2.0, 1.0, 1024),
         ("plain", 90, 1.0, 0.0, 512), ("plain", 120, 1.0, 0.0, 1024), ("antisym", 10, 1.0, 0.0, 1024)]
CHECKPOINTS = [0, 1, 2, 4, 8, 16, 32]


def _job(case):
    arch, base, tau, rho, N = case
    torch.set_num_threads(1)
    xp, yp, xt, yt = C.data()
    kern = {(r["arch"], r["base"], r["tau"], r["rho"]): r for r in json.load(open(os.path.join(RESULTS, "C_kernel.json")))}
    r0 = kern[(arch, base, tau, rho)]
    m = C.load_model(arch, base, r0["lam_b"])
    lr = 0.125 * C.B_RL / r0["k_d"]
    out, done = [], 0
    torch.manual_seed(0)
    for ep in CHECKPOINTS:
        if ep > done:
            majority_vote_rl(m, xp[:N], xt, yp[:N], yt, tau=tau, B=C.B_RL, lr=lr, epochs=ep - done, seed=ep, n_log=2)
            done = ep
        ks = kernel_stats(m, xp[:256], tau)
        z = ks.pop("z")
        kdp = ks["k_d"] - ks["k_u"] - ks["k_s"]
        with torch.no_grad():
            ztr = (m(xp[:N]) / tau).numpy()
        out.append(dict(epoch=ep, ku_kdp=ks["k_u"] / kdp, ks_kdp=ks["k_s"] / kdp, k_d=ks["k_d"],
                        phi=float(abs(np.sign(ztr).mean())), med_abs_z=float(np.median(np.abs(ztr)))))
    return dict(case=list(case), traj=out)


if __name__ == "__main__":
    with Pool(len(CASES)) as p:
        res = p.map(_job, CASES)
    json.dump(res, open(os.path.join(RESULTS, "C_drift.json"), "w"), indent=1)
    for r in res:
        print(r["case"])
        for x in r["traj"]:
            print("   ep %2d  ku/kd' %.3f  ks/kd' %.3f  phi %.2f  med|z| %.2f" % (x["epoch"], x["ku_kdp"], x["ks_kdp"], x["phi"], x["med_abs_z"]))
