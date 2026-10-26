"""Beyond the uniform mode: which collective kernel modes exist, and what rule does a
remedied (balanced) but failed run converge to?"""
import json
import os

import numpy as np
import torch

import stage_c as C
from common import RESULTS
from rlvr_collapse.nn import kernel_stats, majority_vote_rl, per_example_grads


def mode_spectrum(arch, base, tau, lam_b=0.0, n=384):
    xp, _, _, _ = C.data()
    m = C.load_model(arch, base, lam_b)
    with torch.no_grad():
        z = (m(xp[:n]) / tau).double()
    J = per_example_grads(m, xp[:n], tau).double()
    K = J @ J.T
    s = torch.sign(z)
    one = torch.ones(n, dtype=torch.float64)
    # projector removing the prediction-aligned direction s and the uniform direction 1
    Q, _ = torch.linalg.qr(torch.stack([s, one], 1))
    P = torch.eye(n, dtype=torch.float64) - Q @ Q.T
    ev_rest = torch.linalg.eigvalsh(P @ K @ P).flip(0)[:5]
    u = one / np.sqrt(n)
    return dict(k_d=float(K.diagonal().mean()), uniform=float(u @ K @ u), rest_top5=ev_rest.tolist(),
                s_mode=float((s / np.sqrt(n)) @ K @ (s / np.sqrt(n))))


def shortcut_rule(beta=1.0, N=512, seed=0):
    xp, yp, xt, yt = C.data()
    kern = {(r["arch"], r["base"], r["tau"], r["rho"]): r for r in json.load(open(os.path.join(RESULTS, "C_kernel.json")))}
    r0 = kern[("plain", 90, 2.0, 0.0)]
    m = C.load_model("plain", 90)
    torch.set_num_threads(4)
    majority_vote_rl(m, xp[:N], xt, yp[:N], yt, tau=2.0, B=C.B_RL, lr=0.125 * C.B_RL / r0["k_d"], epochs=C.EPOCHS,
                     seed=seed, n_log=2, beta=beta)
    with torch.no_grad():
        pred = (m(xt) > 0).numpy()
    x = xt.numpy()
    feats = {"true rule": yt.numpy() == 1}
    for k in range(6):
        feats[f"x{k + 1}>=5"] = x[:, k] >= 5
        feats[f"x{k + 1} odd"] = x[:, k] % 2 == 1
    for i in range(3):
        for j in range(3, 6):
            feats[f"x{i + 1}>x{j + 1}"] = x[:, i] > x[:, j]
    feats["x1+x2>x4+x5"] = x[:, 0] + x[:, 1] > x[:, 3] + x[:, 4]
    agree = {k: float(max(np.mean(pred == v), np.mean(pred != v))) for k, v in feats.items()}
    return dict(sorted(agree.items(), key=lambda kv: -kv[1])[:6])


if __name__ == "__main__":
    out = {}
    for arch, base, tau in [("plain", 90, 2.0), ("plain", 90, 1.0), ("plain", 120, 1.0), ("antisym", 20, 1.0)]:
        sp = mode_spectrum(arch, base, tau)
        out[f"{arch}{base}_tau{tau}"] = sp
        print(arch, base, tau, "k_d %.1f  uniform-mode %.1f  s-mode %.1f  top other modes %s" %
              (sp["k_d"], sp["uniform"], sp["s_mode"], np.round(sp["rest_top5"], 1)), flush=True)
    out["shortcut_beta1_N512"] = shortcut_rule()
    print("remedied run (beta=1, N=512): agreement of predictions with simple rules:", out["shortcut_beta1_N512"])
    json.dump(out, open(os.path.join(RESULTS, "C_modes.json"), "w"), indent=1)
