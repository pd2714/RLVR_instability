"""Stage C: a small Transformer trained with majority-vote (label-free) RL.

Task: x in {0..9}^6, answer A iff x1+x2+x3 > x4+x5+x6 (ties excluded).  The
policy is a 2-way softmax read out from a CLS token; z(x) = logit_A - logit_B.
An explicit scalar output bias b enters as z = f_theta(x) + lam_b * b, so
lam_b adds a controllable amount of prediction-agnostic shared coupling on top
of the network's intrinsic coupling.  With antisym=True the readout is
z(x) = f(x) - f(swap(x)), which respects the task symmetry (swapping the two
halves flips the answer) and removes the input-independent shared direction.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from .sim import expected_majority_reward

SEQ, VOCAB = 6, 10
CLS = VOCAB


class TinyTransformer(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, lam_b=0.0, antisym=False):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 1, d)
        self.pos = nn.Parameter(0.02 * torch.randn(SEQ + 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(()))
        self.lam_b = lam_b
        self.antisym = antisym

    def readout(self, x):
        cls = torch.full((x.shape[0], 1), CLS, dtype=x.dtype, device=x.device)
        h = self.emb(torch.cat([cls, x], 1)) + self.pos
        h = self.ln(self.enc(h))
        return self.head(h[:, 0]).squeeze(-1)

    def forward(self, x):
        z = self.readout(x)
        if self.antisym:  # z(swap x) = -z(x): removes the trivial input-independent shared direction
            z = z - self.readout(torch.cat([x[:, 3:], x[:, :3]], 1))
        return z + self.lam_b * self.b


def make_data(n, rng):
    xs, ys = [], []
    while sum(len(x) for x in xs) < n:
        x = rng.integers(0, VOCAB, size=(2 * n, SEQ))
        d = x[:, :3].sum(1) - x[:, 3:].sum(1)
        keep = d != 0
        xs.append(x[keep])
        ys.append((d[keep] > 0).astype(np.int64))
    x = np.concatenate(xs)[:n]
    y = np.concatenate(ys)[:n]
    return torch.tensor(x), torch.tensor(y)


def pretrain(model, steps, seed=0, lr=1e-3, batch=128, log_every=100, x_eval=None, y_eval=None):
    """Supervised warm-up (stands in for the base model's prior ability)."""
    rng = np.random.default_rng(seed)
    x, y = make_data(steps * batch, rng)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for t in range(steps):
        xb, yb = x[t * batch:(t + 1) * batch], y[t * batch:(t + 1) * batch]
        z = model(xb)
        loss = nn.functional.binary_cross_entropy_with_logits(z, yb.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if x_eval is not None and (t + 1) % log_every == 0:
            with torch.no_grad():
                ze = model(x_eval)
            hist.append((t + 1, float(((ze > 0).long() == y_eval).float().mean())))
    return hist


def per_example_grads(model, x, tau=1.0, chunk=128):
    """Matrix of gradients of z(x)/tau w.r.t. all parameters, shape (n, P)."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def zfun(p, xi):
        return functional_call(model, p, (xi[None],))[0] / tau

    g = vmap(grad(zfun), in_dims=(None, 0))
    rows = []
    for i in range(0, len(x), chunk):
        gi = g(params, x[i:i + chunk])
        rows.append(torch.cat([gi[k].reshape(len(x[i:i + chunk]), -1) for k in params], 1))
    return torch.cat(rows, 0)


def kernel_stats(model, x, tau=1.0):
    """Empirical NTK decomposition on problems x (at temperature tau).

    k_d : mean self-kernel K_ii
    k_u : mean off-diagonal K_ij           (prediction-agnostic shared coupling)
    k_s : mean off-diagonal s_i s_j K_ij    (coupling aligned with current predictions)
    """
    with torch.no_grad():
        z = model(x) / tau
    J = per_example_grads(model, x, tau)
    K = (J @ J.T).double()
    n = len(x)
    s = torch.sign(z).double()
    off = ~torch.eye(n, dtype=torch.bool)
    k_d = float(K.diagonal().mean())
    k_u = float(K[off].mean())
    k_s = float((K * s[:, None] * s[None, :])[off].mean())
    return dict(k_d=k_d, k_u=k_u, k_s=k_s, z=z.numpy())


def majority_vote_rl(model, x_train, x_test, y_train, y_test, *, tau=1.0, K=3, B=32, lr=0.05, epochs=50,
                     seed=0, n_log=40, beta=0.0):
    """Label-free RL: REINFORCE with majority-vote rewards, plain SGD.
    beta > 0 adds the batch marginal-entropy bonus beta * H(mean_batch sigmoid(z))."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    N = len(x_train)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    steps_per_epoch = max(N // B, 1)
    T = int(epochs * steps_per_epoch)
    log_every = max(T // n_log, 1)
    hist = {k: [] for k in ["step", "epoch", "phi", "acc", "test_acc", "test_phi", "reward", "conf"]}

    def log(t):
        with torch.no_grad():
            z = model(x_train) / tau
            zt = model(x_test) / tau
        p = torch.sigmoid(z)
        hist["step"].append(t)
        hist["epoch"].append(t / steps_per_epoch)
        hist["phi"].append(float(torch.sign(z).mean().abs()))
        hist["acc"].append(float(((z > 0).long() == y_train).float().mean()))
        hist["test_acc"].append(float(((zt > 0).long() == y_test).float().mean()))
        hist["test_phi"].append(float(torch.sign(zt).mean().abs()))
        hist["reward"].append(float(expected_majority_reward(p.numpy(), K).mean()))
        hist["conf"].append(float(z.abs().mean()))

    perm = None
    for t in range(T):
        j = t % steps_per_epoch
        if j == 0:
            perm = torch.randperm(N, generator=gen)
        if t % log_every == 0:
            log(t)
        idx = perm[j * B:(j + 1) * B]
        z = model(x_train[idx]) / tau
        p = torch.sigmoid(z).detach()
        y = (torch.rand(len(idx), K, generator=gen) < p[:, None]).float()
        k = y.sum(1)
        tie = (torch.rand(len(idx), generator=gen) < 0.5).float()
        winA = torch.where(2 * k > K, 1.0, torch.where(2 * k < K, 0.0, tie))
        r = (y == winA[:, None]).float()
        g = (r * (y - p[:, None])).mean(1)               # expected-gradient weight on z_i
        loss = -(g * z).sum() / len(idx)
        if beta:
            pbar = torch.sigmoid(z).mean().clamp(1e-6, 1 - 1e-6)
            loss = loss - beta * -(pbar * pbar.log() + (1 - pbar) * (1 - pbar).log())
        opt.zero_grad()
        loss.backward()
        opt.step()
    log(T)
    return {k: np.array(v) for k, v in hist.items()}
