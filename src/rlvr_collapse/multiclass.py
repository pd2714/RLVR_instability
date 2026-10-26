"""Multi-answer (M candidate answers, softmax policy) version of the model.

Problem i has logits z_i = u_i + lam * v in R^M, with a shared vector v.
Theory: exact expected gradient g(z) by enumerating multinomial count
vectors; collective growth rate from the permutation-averaged Jacobian
restricted to the zero-sum (softmax-identifiable) subspace.
"""
from functools import lru_cache
from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .theory import sigmoid


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


@lru_cache(maxsize=None)
def _compositions(K, M):
    if M == 1:
        return [(K,)]
    return [(k,) + rest for k in range(K + 1) for rest in _compositions(K - k, M - 1)]


@lru_cache(maxsize=None)
def _tables(K, M, adv):
    """Count vectors n, multinomial coefficients, and per-class coefficient
    c_c = (1/K) n_c * E_tiebreak[A_c] so that E-grad = sum_n P(n) (c - sum(c) p)."""
    comps = np.array(_compositions(K, M))
    mult = np.array([factorial(K) / np.prod([factorial(x) for x in n]) for n in comps])
    coef = np.zeros(comps.shape, float)
    for row, n in enumerate(comps):
        winners = np.flatnonzero(n == n.max())
        for w in winners:
            # every sample with answer c gets reward 1[c == w]; advantage per class
            rc = (np.arange(M) == w).astype(float)
            rbar = n[w] / K
            sd = np.sqrt(rbar * (1 - rbar))
            if adv == "reinforce":
                Ac = rc
            elif adv == "mean":
                Ac = rc - rbar
            elif adv == "grpo":
                Ac = (rc - rbar) / sd if sd > 0 else 0 * rc
            else:
                raise ValueError(adv)
            coef[row] += n * Ac / len(winners) / K
    return comps, mult, coef


def expected_grad(z, K=3, adv="reinforce"):
    """z: (..., M) logits -> expected per-sample gradient (..., M)."""
    z = np.asarray(z, float)
    M = z.shape[-1]
    comps, mult, coef = _tables(K, M, adv)
    p = softmax(z)
    logp = np.log(np.clip(p, 1e-300, None))
    P = mult * np.exp(logp @ comps.T)                 # (..., ncomp)
    c = P @ coef                                       # (..., M)
    return c - (P @ coef.sum(1))[..., None] * p


def local_drift(a, M, K=3, adv="reinforce"):
    """Drift of the margin a for logits a*e_0 (all other logits equal)."""
    z = np.zeros(M)
    z[0] = a
    g = expected_grad(z, K, adv)
    return g[0] - g[1]


def collective_rate(a, M, K=3, adv="reinforce", h=1e-5):
    """mu(a): eigenvalue of the label-averaged Jacobian on the zero-sum subspace
    (the shared-mode growth rate per unit lambda^2)."""
    J = np.zeros((M, M))
    for s in range(M):
        z0 = np.zeros(M)
        z0[s] = a
        for j in range(M):
            e = np.zeros(M)
            e[j] = h
            J[:, j] += (expected_grad(z0 + e, K, adv) - expected_grad(z0 - e, K, adv)) / (2 * h) / M
    P = np.eye(M) - 1.0 / M
    return float(np.trace(P @ J @ P) / (M - 1))


def critical_margin(M, K=3, adv="reinforce"):
    return brentq(lambda a: collective_rate(a, M, K, adv), 1e-3, 8.0)


def amplification(a0, M, K=3, adv="reinforce"):
    """G(a0) = int_{a0}^{a*} mu(a)/f(a) da;  Lambda = lambda^2 N G."""
    a_star = critical_margin(M, K, adv)
    if a0 >= a_star:
        return 0.0
    val, _ = quad(lambda a: collective_rate(a, M, K, adv) / local_drift(a, M, K, adv), a0, a_star, limit=200)
    return float(val)


def run_multiclass(cfg):
    """Stochastic simulation; mirrors sim.run for M > 2 answers."""
    rng = np.random.default_rng(cfg.seed)
    R, N, B, K, M = cfg.runs, cfg.N, cfg.B, cfg.K, cfg.M
    a = float(cfg.a)
    s = np.tile(np.arange(N) % M, (R, 1))
    s = rng.permuted(s, axis=1)
    u = a * (s[..., None] == np.arange(M)).astype(float)
    other = (s + rng.integers(1, M, size=(R, N))) % M
    correct = np.where(rng.random((R, N)) < cfg.q_correct, s, other)
    v = np.zeros((R, M))
    if cfg.opt == "adam":
        mu, su = np.zeros_like(u), np.zeros_like(u)
        mv, sv = np.zeros_like(v), np.zeros_like(v)
        b1, b2 = cfg.betas
    steps_per_epoch = N // B
    T = int(round(cfg.epochs * steps_per_epoch))
    log_every = max(T // cfg.n_log, 1)
    rows = np.arange(R)[:, None]
    hist = {k: [] for k in ["step", "epoch", "phi", "acc", "pass1", "reward", "flipped", "conf", "shift"]}

    def log(t):
        z = u + cfg.lam * v[:, None, :]
        p = softmax(z)
        am = z.argmax(-1)
        frac = (am[..., None] == np.arange(M)).mean(1)          # (R, M)
        hist["step"].append(t)
        hist["epoch"].append(t / steps_per_epoch)
        hist["phi"].append((frac.max(1) - 1 / M) / (1 - 1 / M))
        hist["acc"].append(np.mean(am == correct, axis=1))
        hist["pass1"].append(np.mean(np.take_along_axis(p, correct[..., None], -1)[..., 0], axis=1))
        ps = np.sort(p, -1)
        hist["reward"].append(np.mean(ps[..., -1], axis=1))          # proxy: top-answer prob
        hist["flipped"].append(np.mean(am != s, axis=1))
        zs = np.sort(z, -1)
        hist["conf"].append(np.mean(zs[..., -1] - zs[..., -2], axis=1))
        hist["shift"].append(cfg.lam * (v.max(1) - v.min(1)))

    perm = None
    for t in range(T):
        j = t % steps_per_epoch
        if j == 0:
            perm = rng.permuted(np.tile(np.arange(N), (R, 1)), axis=1)
        if t % log_every == 0:
            log(t)
        idx = perm[:, j * B:(j + 1) * B]
        z = u[rows, idx] + cfg.lam * v[:, None, :]                   # (R, B, M)
        p = softmax(z)
        cum = np.cumsum(p, -1)
        y = (rng.random((R, B, K, 1)) > cum[:, :, None, :-1]).sum(-1)  # (R, B, K) sampled answers
        Y = (y[..., None] == np.arange(M)).astype(float)               # one-hot (R, B, K, M)
        counts = Y.sum(2)
        w = (counts + 1e-3 * rng.random(counts.shape)).argmax(-1)       # random tie-break
        r = (y == w[..., None]).astype(float)
        if cfg.adv == "reinforce":
            A = r
        else:
            A = r - r.mean(-1, keepdims=True)
            if cfg.adv == "grpo":
                sd = r.std(-1, keepdims=True)
                A = np.where(sd > 0, A / np.where(sd > 0, sd, 1.0), 0.0)
        g = (A[..., None] * (Y - p[:, :, None, :])).mean(2)            # (R, B, M)
        gu = g / B
        gv = cfg.lam * g.sum(1) / B
        if cfg.opt == "sgd":
            u[rows, idx] += cfg.lr * gu
            v += cfg.lr * gv
        else:
            Gu = np.zeros_like(u)
            Gu[rows, idx] = gu
            mu = b1 * mu + (1 - b1) * Gu
            su = b2 * su + (1 - b2) * Gu**2
            mv = b1 * mv + (1 - b1) * gv
            sv = b2 * sv + (1 - b2) * gv**2
            c1, c2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
            u += cfg.lr * (mu / c1) / (np.sqrt(su / c2) + cfg.eps)
            v += cfg.lr * (mv / c1) / (np.sqrt(sv / c2) + cfg.eps)
    log(T)
    return {k: np.array(vv) for k, vv in hist.items()}
