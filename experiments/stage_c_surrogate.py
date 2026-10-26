"""Parameter-free predictions for Stage C from quantities measured on the initial network.

For each condition we build the kernel-calibrated surrogate of the toy model:
lambda^2 = k_u / k_d', mu^2 = k_s / k_d', the network's own initial logits on the
N training problems, and the same B, K, epochs and per-visit step as the network
run.  We record (1) the stochastic surrogate's collapse probability, (2) the
deterministic kernel mean-field prediction, and (3) the amplification
Lambda(N) = G N k_u / (k_d' + N k_s).
"""
import json
import os

import numpy as np
import torch

from common import RESULTS, SimConfig, run_sweep
import stage_c as C
from rlvr_collapse.theory import kernel_meanfield


def logits(arch, base, tau, lam_b):
    xp, _, _, _ = C.data()
    m = C.load_model(arch, base, lam_b)
    with torch.no_grad():
        return (m(xp[:max(C.NS[arch])]) / tau).numpy()


if __name__ == "__main__":
    torch.set_num_threads(4)
    kern = json.load(open(os.path.join(RESULTS, "C_kernel.json")))
    cfgs, preds = [], []
    for r in kern:
        z = logits(r["arch"], r["base"], r["tau"], r["lam_b"])
        lam = float(np.sqrt(max(r["k_u"], 0.0) / r["k_dp"]))
        mu = float(np.sqrt(max(r["k_s"], 0.0) / r["k_dp"]))
        tau_max = C.EPOCHS * 0.125 * r["k_dp"] / r["k_d"]
        for N in C.NS[r["arch"]]:
            B = min(C.B_RL, N)
            c = SimConfig(N=N, B=B, K=3, lam=lam, mu=mu, z0=z[:N], lr=0.125 * B / (1 + lam**2 + mu**2),
                          epochs=C.EPOCHS, runs=24, n_log=50)
            c.extra = dict(arch=r["arch"], base=r["base"], tau=r["tau"], rho=r["rho"])
            cfgs.append(c)
            _, phi, _ = kernel_meanfield(z[:N], r["k_dp"], max(r["k_u"], 0.0), r["k_s"], tau_max)
            Lam = r["G"] * N * max(r["k_u"], 0.0) / (r["k_dp"] + N * r["k_s"])
            preds.append(dict(arch=r["arch"], base=r["base"], tau=r["tau"], rho=r["rho"], N=N,
                              lam2=lam**2, mu2=mu**2, kmf_phi=float(phi[-1]), Lambda=float(Lam)))
    json.dump(preds, open(os.path.join(RESULTS, "C_kmf.json"), "w"), indent=1)
    run_sweep("C_surrogate", cfgs)
