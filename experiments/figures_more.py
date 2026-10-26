"""Figures for batch size / long training / heterogeneity / feature coupling /
remedy (Stage A, cont.) and Stage B robustness."""
import json
import os
import sys

import numpy as np

from common import FIGURES, RESULTS, fit_Nc, load
from figures_a import by, curve
from plotstyle import CAT, COL_W, FULL_W, INK2, MUTED, SHADE, panel_label, plt, seq, setup
from rlvr_collapse import multiclass
from rlvr_collapse.theory import amplification_heterogeneous, amplification_per_coupling, critical_logit

setup()
SUMMARY = {}
S_A = json.load(open(os.path.join(RESULTS, "summary_a.json")))
LC = S_A["Lambda_c"]
G5 = amplification_per_coupling(0.5)


def nc_or_nan(rs):
    Ns, pc, n = curve(rs)
    if pc.max() < 0.5 or pc.min() > 0.5:
        return np.nan
    return fit_Nc(Ns, pc, n)[0]


def fig_batch_stable():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 1.8))
    # (a) N_c vs batch size, fixed step and linear scaling
    ax = axs[0]
    out = {}
    for name, c, mk, lab in [("A_batch", CAT[0], "o", r"fixed step $\eta$"), ("A_batch_linear", CAT[1], "s", r"$\eta\propto B$")]:
        R = by(load(name), "B")
        Bs = sorted(R)
        ncs = [nc_or_nan(R[b]) for b in Bs]
        out[name] = dict(zip(map(int, Bs), map(float, ncs)))
        ax.plot(Bs, ncs, mk + "-", color=c, label=lab)
    Bg = np.geomspace(8, 128, 50)
    nc32 = out["A_batch"].get(32)
    ax.plot(Bg, nc32 + 0.5 * np.log(Bg / 32) / (0.01 * G5), color=INK2, ls="--", lw=0.8, label=r"theory: $+\frac{\ln B}{2\lambda^2G}$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size $B$ (problems/step)")
    ax.set_ylabel(r"$N_c$")
    ax.set_ylim(0, 1.6 * np.nanmax([v for d in out.values() for v in d.values()]))
    ax.legend(loc="upper left", fontsize=5.8)
    panel_label(ax, "a")
    SUMMARY["Nc_vs_B"] = out

    # (b) long training: confident models never collapse
    ax = axs[1]
    R = load("A_stable")
    rows = []
    for r in R:
        c = r["cfg"]
        rows.append((c["a"], c["lam"], c["N"], c["epochs"], r["p_collapse"], np.mean(r["final"]["flipped"]), np.mean(r["final"]["phi"])))
    SUMMARY["stable"] = rows
    conf = [x for x in rows if x[0] > 1.32]
    unc = [x for x in rows if x[0] < 1.32]
    for j, a in enumerate(sorted({x[0] for x in conf})):
        xs = [x for x in conf if x[0] == a]
        Lam = [x[1] ** 2 * x[2] for x in xs]
        ax.scatter(np.array(Lam) * 1.12 ** (j - 1), [x[6] for x in xs], color=seq(3)[j], marker="os^"[j], s=14,
                   label=f"a={a}", zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\lambda^2N$")
    ax.set_ylabel(r"final $\Phi$ (10$\times$ longer training)")
    ax.legend(loc="center left", fontsize=5.8)
    panel_label(ax, "b")
    # trajectories of the shift for a confident model at the largest lambda^2 N
    ax = axs[2]
    big = max([r for r in R if r["cfg"]["a"] > 1.32], key=lambda r: r["cfg"]["lam"] ** 2 * r["cfg"]["N"] * (r["cfg"]["a"] < 1.5))
    uns = [r for r in R if r["cfg"]["a"] < 1.32]
    for r, c, lab in [(big, CAT[0], f"a={big['cfg']['a']}, N={big['cfg']['N']}, $\\lambda$={big['cfg']['lam']}")] + \
                      [(u, CAT[1 + k], f"a={u['cfg']['a']}, N={u['cfg']['N']}, $\\lambda$={u['cfg']['lam']}") for k, u in enumerate(uns)]:
        ep = np.array(r["traj"]["epoch"])
        ax.plot(ep, r["traj"]["phi"], color=c, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"mean $\Phi$")
    ax.set_ylim(-0.02, 0.5)
    ax.legend(loc="upper right", fontsize=5.5)
    panel_label(ax, "c")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGURES, "fig_batch_stable.pdf"))


def fig_hetero_feature_remedy():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 1.8))
    # (a) heterogeneous initial confidence: P(collapse) vs predicted Lambda
    ax = axs[0]
    R = by(load("A_hetero"), "profile")
    het = {}
    for j, prof in enumerate(sorted(R)):
        rs = R[prof]
        G = rs[0]["cfg"]["extra"]["G_pop"]
        Ns, pc, n = curve(rs)
        ax.plot(0.01 * Ns * G, pc, "o-", color=CAT[j % 8], ms=2.5, lw=0.9, label=prof)
        het[prof] = dict(G=G, Nc=float(nc_or_nan(rs)))
    ax.axvline(LC, color=MUTED, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"predicted $\Lambda=\lambda^2NG[\rho_0]$")
    ax.set_ylabel("P(collapse)")
    ax.legend(fontsize=5, loc="upper left", ncol=1)
    panel_label(ax, "a")
    SUMMARY["hetero"] = het

    # (b) feature coupling: N_c saturates / diverges
    ax = axs[1]
    R = by(load("A_feature"), "mu")
    mus = sorted(R)
    feat = {}
    for j, mu in enumerate(mus):
        Ns, pc, n = curve(R[mu])
        ax.plot(Ns, pc, "o-", color=seq(len(mus))[j], label=f"{mu:g}", ms=2.5)
        feat[mu] = float(nc_or_nan(R[mu]))
    ax.set_xscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("P(collapse)")
    ax.legend(title=r"$\mu$", fontsize=5.5, title_fontsize=6, loc="center right")
    panel_label(ax, "b")
    SUMMARY["feature_Nc"] = feat
    ins = ax.inset_axes([0.08, 0.5, 0.36, 0.42])
    mg = np.linspace(0, 0.07, 100)
    pred = 1 / (0.01 * G5 / LC - mg**2)
    pred[pred < 0] = np.inf
    ins.plot(mg, pred, color=INK2, lw=0.8, ls="--")
    mu_arr = np.array(mus)
    nc_arr = np.array([feat[m] for m in mus])
    ins.plot(mu_arr, np.where(np.isnan(nc_arr), 6e4, nc_arr), "o", color=CAT[0], ms=2.5)
    ins.set_yscale("log")
    ins.set_ylim(300, 8e4)
    ins.set_xticks([0, 0.04])
    ins.tick_params(labelsize=5)
    ins.set_xlabel(r"$\mu$", fontsize=5.5, labelpad=0)
    ins.set_ylabel(r"$N_c$", fontsize=5.5, labelpad=0)

    # (c) remedy: marginal-entropy bonus
    ax = axs[2]
    R = load("A_remedy")
    rem = {}
    for j, a in enumerate([0.2, 0.5]):
        for k, N in enumerate([512, 2048, 8192]):
            rs = sorted([r for r in R if r["cfg"]["a"] == a and r["cfg"]["N"] == N], key=lambda r: r["cfg"]["beta"])
            betas = [r["cfg"]["beta"] for r in rs]
            pc = [r["p_collapse"] for r in rs]
            acc = [np.mean(r["final"]["acc"]) for r in rs]
            rem[f"a={a},N={N}"] = dict(beta=betas, p_collapse=pc, acc=acc)
            ax.plot(betas, acc, ["o-", "s--"][j], color=seq(3)[k], ms=2.5, lw=0.9,
                    label=f"N={N}" if j == 1 else None)
    from rlvr_collapse.theory import dF, sigmoid
    for j, a in enumerate([0.2, 0.5]):
        p = sigmoid(a)
        ax.axvline(dF(a) / (4 * (p * (1 - p)) ** 2), color=MUTED, ls=[":", "-."][j], lw=0.8)
    ax.plot([], [], "o-", color=INK2, label="a=0.2")
    ax.plot([], [], "s--", color=INK2, label="a=0.5")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel(r"marginal-entropy weight $\beta$")
    ax.set_ylabel("final accuracy")
    ax.legend(fontsize=5.5, loc="lower right")
    panel_label(ax, "c")
    SUMMARY["remedy"] = rem
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGURES, "fig_hetero_feature_remedy.pdf"))


def fig_robust():
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 1.8))
    ax = axs[0]
    R = load("B_grpo")
    groups = {}
    for r in R:
        groups.setdefault((r["cfg"]["adv"], r["cfg"]["K"]), []).append(r)
    rob = {}
    for j, key in enumerate(sorted(groups)):
        rs = sorted(groups[key], key=lambda r: r["cfg"]["N"])
        G = rs[0]["cfg"]["extra"]["G"]
        Ns, pc, n = curve(rs)
        ax.plot(0.01 * Ns * G, pc, "o-", color=CAT[j], ms=2.5, lw=0.9, label=f"{key[0]} K={key[1]}")
        nc = nc_or_nan(rs)
        rob[f"{key[0]}_K{key[1]}"] = dict(G=G, Nc=float(nc), Lambda_c=float(0.01 * nc * G))
    ax.axvline(LC, color=MUTED, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Lambda=\lambda^2NG_{K}$")
    ax.set_ylabel("P(collapse)")
    ax.legend(fontsize=5, loc="upper left")
    panel_label(ax, "a")

    ax = axs[1]
    R = by(load("B_multi"), "M")
    for j, M in enumerate(sorted(R)):
        rs = R[M]
        G = rs[0]["cfg"]["extra"]["G"]
        Ns, pc, n = curve(rs)
        ax.plot(0.01 * Ns * G, pc, "o-", color=CAT[j], ms=2.5, lw=0.9, label=f"M={M}, K={rs[0]['cfg']['K']}")
        nc = nc_or_nan(rs)
        rob[f"multi_M{M}"] = dict(G=G, Nc=float(nc), Lambda_c=float(0.01 * nc * G))
    ax.axvline(LC, color=MUTED, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Lambda=\lambda^2NG_{M}$")
    ax.legend(fontsize=5.5, loc="upper left")
    panel_label(ax, "b")

    ax = axs[2]
    R = by(load("B_adam"), "lam")
    lams = sorted(R)
    nc_adam = []
    for j, lam in enumerate(lams):
        Ns, pc, n = curve(R[lam])
        ax.plot(Ns, pc, "o-", color=seq(len(lams))[j], ms=2.5, lw=0.9, label=f"{lam:g}")
        nc_adam.append(nc_or_nan(R[lam]))
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (Adam)")
    ax.set_ylabel("P(collapse)")
    ax.legend(title=r"$\lambda$", fontsize=5.5, title_fontsize=6, loc="lower right")
    panel_label(ax, "c")
    ok = ~np.isnan(nc_adam)
    if ok.sum() >= 2:
        sl = np.polyfit(np.log(np.array(lams)[ok]), np.log(np.array(nc_adam)[ok]), 1)[0]
        rob["adam_slope"] = float(sl)
    rob["adam_Nc"] = dict(zip(map(float, lams), map(float, nc_adam)))
    SUMMARY["robust"] = rob
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGURES, "fig_robust.pdf"))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[f"fig_{name}"]()
    path = os.path.join(RESULTS, "summary_more.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(SUMMARY)
    json.dump(old, open(path, "w"), indent=1, default=float)
    print(json.dumps(SUMMARY, indent=1, default=float)[:3000])
