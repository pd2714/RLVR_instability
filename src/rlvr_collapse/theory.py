"""Exact single-problem drift and mean-field theory for majority-vote RLVR.

Binary-answer problem with logit z (P(A) = sigmoid(z)).  K responses are
sampled, the majority answer gets reward 1 (ties broken uniformly at random),
and the policy-gradient estimator uses one of three advantages:

  'reinforce' : A_j = r_j                       (no baseline)
  'mean'      : A_j = r_j - mean(r)             (group-mean baseline, Dr.GRPO)
  'grpo'      : A_j = (r_j - mean(r)) / std(r)  (GRPO; 0 if std == 0)

F(z) is the expected per-sample gradient on z, averaged over the K samples.
"""
from functools import lru_cache
from math import comb

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@lru_cache(maxsize=None)
def _count_table(K, adv):
    """For k = #A among K samples, return (c_A, c_B) such that the summed
    gradient sum_j A_j (y_j - p) = c_A * (1 - p) + c_B * (-p), averaged over
    tie-breaks and divided by K."""
    cA = np.zeros(K + 1)
    cB = np.zeros(K + 1)
    for k in range(K + 1):
        winners = ["A"] if 2 * k > K else (["B"] if 2 * k < K else ["A", "B"])
        for w in winners:
            r = np.array([1.0] * k + [0.0] * (K - k)) if w == "A" else np.array([0.0] * k + [1.0] * (K - k))
            if adv == "reinforce":
                A = r
            elif adv == "mean":
                A = r - r.mean()
            elif adv == "grpo":
                s = r.std()
                A = (r - r.mean()) / s if s > 0 else 0.0 * r
            else:
                raise ValueError(adv)
            cA[k] += A[:k].sum() / len(winners) / K
            cB[k] += A[k:].sum() / len(winners) / K
    return cA, cB


def F(z, K=3, adv="reinforce"):
    """Expected per-sample policy-gradient drift on the logit z."""
    z = np.asarray(z, dtype=float)
    p = sigmoid(z)
    cA, cB = _count_table(K, adv)
    out = np.zeros_like(p)
    for k in range(K + 1):
        pk = comb(K, k) * p**k * (1 - p) ** (K - k)
        out = out + pk * (cA[k] * (1 - p) - cB[k] * p)
    return out


def dF(z, K=3, adv="reinforce", h=1e-5):
    return (F(np.asarray(z) + h, K, adv) - F(np.asarray(z) - h, K, adv)) / (2 * h)


@lru_cache(maxsize=None)
def critical_logit(K=3, adv="reinforce"):
    """z* > 0 where F'(z*) = 0: edge of the unstable (uncertain) window."""
    return brentq(lambda x: float(dF(x, K, adv)), 1e-3, 6.0)


def F_star(K=3, adv="reinforce"):
    return float(F(critical_logit(K, adv), K, adv))


def amplification_per_coupling(a, K=3, adv="reinforce"):
    """ln[F*/F(a)]: total log-amplification of the shared mode divided by
    lambda^2 N, for a symmetric population with initial |z| = a < z*."""
    a = np.asarray(a, dtype=float)
    val = np.log(F_star(K, adv) / F(np.minimum(a, critical_logit(K, adv)), K, adv))
    return np.where(a < critical_logit(K, adv), val, 0.0)


def predicted_Nc(a, lam, delta, K=3, adv="reinforce"):
    """N_c = ln(1/delta) / (lambda^2 ln[F*/F(a)])."""
    g = amplification_per_coupling(a, K, adv)
    with np.errstate(divide="ignore"):
        return np.where(g > 0, np.log(1.0 / delta) / (lam**2 * g), np.inf)


def instability_index(z, K=3, adv="reinforce"):
    """Mean F'(z_i) over the population: sign gives (in)stability of the
    symmetric state along the shared direction."""
    return float(np.mean(dF(np.asarray(z), K, adv)))


def amplification_heterogeneous(z0, K=3, adv="reinforce", s_max=400.0, n_grid=4000):
    """General amplification G for an arbitrary set of initial logits z0.

    Each problem sharpens locally, dz/ds = F(z) with s = eta t / N.  The shared
    mode grows at rate lambda^2 N <F'(z(s))> in the same units.  Returns
    G = max_s int_0^s <F'(z(s'))> ds', so that Lambda = lambda^2 N G.
    """
    z0 = np.asarray(z0, dtype=float)
    sol = solve_ivp(lambda s, z: F(z, K, adv), (0, s_max), z0, dense_output=True, rtol=1e-6, atol=1e-9)
    s = np.linspace(0, s_max, n_grid)
    Z = sol.sol(s)
    rate = dF(Z, K, adv).mean(axis=0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(s))])
    return float(max(cum.max(), 0.0))


def meanfield_two_population(N, lam, a, delta, K=3, adv="reinforce", t_max=None):
    """Deterministic mean-field ODE for N/2 problems at +a and N/2 at -a.

    dz_+/dt = F(z_+)/N + lam^2 <F>,  dz_-/dt = F(z_-)/N + lam^2 <F>,
    with a shared initial offset delta (eta = 1).  Returns final (z_+, z_-)
    and whether the minority population flipped sign (collapse).
    """
    if t_max is None:
        t_max = 60.0 * N / max(F_star(K, adv), 1e-6) + 50.0 / lam**2

    def rhs(t, y):
        f = F(y, K, adv)
        return f / N + lam**2 * f.mean()

    def flipped(t, y):
        return y[1]
    flipped.terminal = True

    sol = solve_ivp(rhs, (0, t_max), [a + delta, -a + delta], events=flipped, rtol=1e-8, atol=1e-12)
    return sol.y[:, -1], sol.status == 1


def kernel_meanfield(z0, k_dp, k_u, k_s, tau_max, K=3, adv="reinforce", n_out=200):
    """Kernel mean-field prediction for a linearized network (no fitted constants).

    Integrates  dz_i/dtau = F(z_i) + (N k_s/k_dp) s_i <s F> + (N k_u/k_dp) <F>
    from the measured initial logits z0 of the N training problems, with
    s = sign(z0) and K ~ k_dp I + k_u 11^T + k_s s s^T.  tau is measured in units
    of the per-problem (self-kernel) sharpening time.  Returns (tau, Phi(tau), z(tau_max)).
    """
    z0 = np.asarray(z0, dtype=float)
    N = len(z0)
    s = np.sign(z0)
    cu, cs = N * k_u / k_dp, N * k_s / k_dp

    def rhs(t, z):
        f = F(z, K, adv)
        return f + cs * s * np.mean(s * f) + cu * np.mean(f)

    t_eval = np.linspace(0, tau_max, n_out)
    sol = solve_ivp(rhs, (0, tau_max), z0, t_eval=t_eval, rtol=1e-6, atol=1e-9, method="LSODA")
    phi = np.abs(np.mean(np.sign(sol.y), axis=0))
    return sol.t, phi, sol.y[:, -1]
