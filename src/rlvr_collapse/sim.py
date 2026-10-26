"""Stochastic simulation of majority-vote RLVR on the minimal multi-problem model.

Binary answers.  Problem i has logit z_i = u_i + lam * v, where u_i is
problem-specific and v is a single shared parameter.  Each optimizer step draws
a minibatch of B problems (epoch-wise shuffling, as a data loader would), samples
K answers per problem, rewards agreement with the sampled majority, and applies
a policy-gradient step on the batch-averaged objective.  R independent runs are
simulated in parallel (one row per run).
"""
from dataclasses import dataclass, field

import numpy as np

from .theory import sigmoid


@dataclass
class SimConfig:
    N: int = 1000           # number of training problems
    lam: float = 0.1        # coupling strength (shared, prediction-agnostic direction)
    beta: float = 0.0       # batch marginal-entropy bonus beta * H(mean_batch p) (anti-collapse remedy)
    mu: float = 0.0         # feature coupling: shared direction aligned with initial predictions s_i (SGD only)
    a: object = 0.5         # initial |z_i| (scalar or array of length N)
    B: int = 32             # problems per minibatch
    K: int = 3              # samples per problem
    M: int = 2              # number of candidate answers (M > 2 -> softmax model)
    lr: float = 4.0         # step size on the batch-averaged objective
    adv: str = "reinforce"  # 'reinforce' | 'mean' | 'grpo'
    opt: str = "sgd"        # 'sgd' | 'adam'
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    epochs: float = 300.0   # passes over the training set
    q_correct: float = 0.8  # P(initial majority answer is correct)
    z0: object = None       # optional explicit signed initial logits (length N); overrides a / balancing
    v0: float = 0.0         # initial shared offset (deterministic seed delta = lam * v0)
    runs: int = 32
    seed: int = 0
    n_log: int = 100
    extra: dict = field(default_factory=dict)


def _advantages(r, adv):
    if adv == "reinforce":
        return r
    m = r.mean(-1, keepdims=True)
    if adv == "mean":
        return r - m
    if adv == "grpo":
        s = r.std(-1, keepdims=True)
        return np.where(s > 0, (r - m) / np.where(s > 0, s, 1.0), 0.0)
    raise ValueError(adv)


def expected_majority_reward(p, K):
    """E[fraction of the K samples that agree with the (tie-broken) majority]."""
    from math import comb
    out = np.zeros_like(p)
    for k in range(K + 1):
        pk = comb(K, k) * p**k * (1 - p) ** (K - k)
        out += pk * max(k, K - k) / K
    return out


def run(cfg: SimConfig):
    if cfg.M > 2:
        from .multiclass import run_multiclass
        return run_multiclass(cfg)
    rng = np.random.default_rng(cfg.seed)
    R, N, B, K = cfg.runs, cfg.N, cfg.B, cfg.K
    assert B <= N
    a = np.broadcast_to(np.asarray(cfg.a, dtype=float), (N,))
    # exactly balanced initial preferences, then which problems are initially right
    if cfg.z0 is not None:  # e.g. measured logits of a network (kernel-calibrated surrogate)
        z0 = np.asarray(cfg.z0, dtype=float)
        s = np.tile(np.where(z0 >= 0, 1.0, -1.0), (R, 1))
        u = np.tile(z0, (R, 1))
    else:
        s = np.where(np.arange(N) < N // 2, 1.0, -1.0)
        s = np.tile(s, (R, 1))
        s = rng.permuted(s, axis=1)
        u = s * a[None, :]
    correct = np.where(rng.random((R, N)) < cfg.q_correct, s, -s)
    v = np.full(R, float(cfg.v0))
    w = np.zeros(R)  # feature-coupling parameter: z_i += mu * s_i * w
    assert cfg.mu == 0 or cfg.opt == "sgd"

    if cfg.opt == "adam":
        mu, su = np.zeros_like(u), np.zeros_like(u)
        mv, sv = np.zeros_like(v), np.zeros_like(v)
        b1, b2 = cfg.betas

    steps_per_epoch = N // B
    T = int(round(cfg.epochs * steps_per_epoch))
    log_every = max(T // cfg.n_log, 1)
    rows = np.arange(R)[:, None]
    hist = {k: [] for k in ["step", "epoch", "phi", "m", "shift", "acc", "pass1", "reward", "flipped", "conf"]}

    def log(t):
        z = u + cfg.lam * v[:, None] + cfg.mu * s * w[:, None]
        p = sigmoid(z)
        sg = np.sign(z)
        hist["step"].append(t)
        hist["epoch"].append(t / steps_per_epoch)
        hist["m"].append(np.mean(2 * p - 1, axis=1))
        hist["phi"].append(np.abs(np.mean(sg, axis=1)))
        hist["shift"].append(cfg.lam * v.copy())
        hist["acc"].append(np.mean(sg == correct, axis=1))
        hist["pass1"].append(np.mean(np.where(correct > 0, p, 1 - p), axis=1))
        hist["reward"].append(np.mean(expected_majority_reward(p, K), axis=1))
        hist["flipped"].append(np.mean(sg != s, axis=1))
        hist["conf"].append(np.mean(np.abs(z), axis=1))

    perm = None
    for t in range(T):
        j = t % steps_per_epoch
        if j == 0:
            perm = rng.permuted(np.tile(np.arange(N), (R, 1)), axis=1)
        if t % log_every == 0:
            log(t)
        idx = perm[:, j * B:(j + 1) * B]                       # (R, B)
        s_b = s[rows, idx]
        z = u[rows, idx] + cfg.lam * v[:, None] + cfg.mu * s_b * w[:, None]
        p = sigmoid(z)
        y = (rng.random((R, B, K)) < p[..., None]).astype(float)  # 1 = answer A
        k = y.sum(-1)
        winA = np.where(2 * k > K, 1.0, np.where(2 * k < K, 0.0, (rng.random((R, B)) < 0.5).astype(float)))
        r = (y == winA[..., None]).astype(float)
        A = _advantages(r, cfg.adv)
        g = (A * (y - p[..., None])).mean(-1)                   # d/dz of per-problem objective
        if cfg.beta:
            pbar = np.clip(p.mean(-1, keepdims=True), 1e-6, 1 - 1e-6)
            g = g + cfg.beta * np.log((1 - pbar) / pbar) * p * (1 - p)
        gu = g / B                                              # gradient of batch mean wrt u_i
        gv = cfg.lam * g.sum(-1) / B
        if cfg.opt == "sgd":
            u[rows, idx] += cfg.lr * gu
            v += cfg.lr * gv
            w += cfg.lr * cfg.mu * (s_b * g).sum(-1) / B
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
