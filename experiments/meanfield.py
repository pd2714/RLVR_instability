"""Deterministic mean-field ODE: exact critical amplification vs the closed form.

In rescaled time s = eta t / N, the two-population mean-field equations read
    dz_pm/ds = F(z_pm) + X <F>,   X = lambda^2 N,
so lambda and N enter only through X: N_c^MF = X_c(a, delta) / lambda^2 exactly.
We find X_c by bisection and compare X_c G(a) with the prediction ln(1/delta).
"""
import json
import os

import numpy as np
from scipy.integrate import solve_ivp

from common import RESULTS
from rlvr_collapse.theory import F, amplification_per_coupling


def collapses(X, a, delta, s_max=200.0):
    def rhs(s, y):
        f = F(y)
        return f + X * f.mean()

    def flipped(s, y):
        return y[1]
    flipped.terminal = True
    sol = solve_ivp(rhs, (0, s_max), [a + delta, -a + delta], events=flipped, method="LSODA", rtol=1e-8, atol=1e-12)
    return sol.status == 1


def X_c(a, delta):
    lo, hi = 1e-2, 1e4
    if not collapses(hi, a, delta):
        return np.inf
    while hi / lo > 1.002:
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if collapses(mid, a, delta) else (mid, hi)
    return float(np.sqrt(lo * hi))


if __name__ == "__main__":
    rows = []
    for a in [0.1, 0.3, 0.5, 0.8, 1.1, 1.25]:
        G = float(amplification_per_coupling(a))
        for delta in [1e-2, 1e-3, 1e-4, 1e-6]:
            xc = X_c(a, delta)
            rows.append(dict(a=a, delta=delta, X_c=xc, G=G, Lambda_c=xc * G, ln_inv_delta=float(np.log(1 / delta))))
            print(f"a={a} delta={delta:g}: X_c={xc:.4g}  Lambda_c=X_c*G={xc * G:.3f}  ln(1/delta)={np.log(1 / delta):.2f}", flush=True)
    with open(os.path.join(RESULTS, "meanfield.json"), "w") as f:
        json.dump(rows, f, indent=1)
