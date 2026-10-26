"""Figures and summary numbers for Stage A (solvable model)."""
import json
import os
import sys

import numpy as np

from common import COLLAPSE_PHI, FIGURES, RESULTS, fit_Nc, load
from plotstyle import CAT, COL_W, FULL_W, INK2, MUTED, SHADE, panel_label, plt, seq, setup
from rlvr_collapse.theory import (F, F_star, amplification_per_coupling, critical_logit, dF)

setup()
os.makedirs(FIGURES, exist_ok=True)
SUMMARY = {}


def by(R, key):
    out = {}
    for r in R:
        k = r["cfg"][key] if key in r["cfg"] else r["cfg"]["extra"][key]
        out.setdefault(k, []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["cfg"]["N"])
    return out


def curve(rs):
    Ns = np.array([r["cfg"]["N"] for r in rs])
    pc = np.array([r["p_collapse"] for r in rs])
    n = np.array([len(r["final"]["phi"]) for r in rs])
    return Ns, pc, n


def fig_mechanism():
    R = by(load("A_lambda"), "lam")[0.1]
    fig, axs = plt.subplots(1, 3, figsize=(FULL_W, 1.85))
    ax = axs[0]
    z = np.linspace(-4, 4, 400)
    zs = critical_logit()
    ax.axvspan(-zs, zs, color=SHADE, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.plot(z, F(z), color=CAT[0], label=r"$F(z)$")
    ax.plot(z, dF(z), color=CAT[1], label=r"$F'(z)$")
    ax.text(0, -0.142, r"unstable window $|z|<1.32$", ha="center", fontsize=6.5, color=INK2)
    ax.set_xlabel(r"logit $z$")
    ax.set_ylim(-0.155, 0.14)
    ax.legend(loc="upper left")
    panel_label(ax, "a")

    ax = axs[1]
    small = R[1]
    big = R[-1]
    Nc, _ = fit_Nc(*curve(list(R)))
    for r, c, rel in [(small, CAT[0], "<"), (big, CAT[1], ">")]:
        ep = np.array(r["traj"]["epoch"])
        ax.plot(ep, r["traj"]["reward"], color=c, ls="-")
        ax.plot(ep, r["traj"]["acc"], color=c, ls="--")
        ax.plot([], [], color=c, lw=2.5, label=f"N={r['cfg']['N']} ({rel} $N_c$)")
    ax.plot([], [], color=INK2, ls="-", label="reward")
    ax.plot([], [], color=INK2, ls="--", label="accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reward / accuracy")
    ax.set_title(f"mean of {len(small['final']['phi'])} seeds, $N_c\\approx{Nc:.0f}$", color=INK2)
    ax.set_xlim(0, 300)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), ncol=2, handlelength=1.6, columnspacing=0.9)
    panel_label(ax, "b")

    ax = axs[2]
    near = min(R, key=lambda r: abs(r["p_collapse"] - 0.5))
    ep = np.array(near["traj"]["epoch"])
    ax.axhline(COLLAPSE_PHI, color=MUTED, lw=0.8, ls=":")
    ax.text(ep[-1], COLLAPSE_PHI + 0.03, "collapse threshold", ha="right", va="bottom", fontsize=6, color=INK2)
    for phi in near["traj_runs"]["phi"]:
        ax.plot(ep, phi, color=CAT[1] if phi[-1] > COLLAPSE_PHI else CAT[0], lw=1.1)
    ax.plot([], [], color=CAT[1], label="collapsed")
    ax.plot([], [], color=CAT[0], label="not collapsed")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"order parameter $\Phi$")
    ax.set_title(f"single seeds at N={near['cfg']['N']} (near $N_c$)", color=INK2)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.75))
    panel_label(ax, "c")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(FIGURES, "fig_mechanism.pdf"))


def fig_scaling():
    RL = by(load("A_lambda"), "lam")
    RA = by(load("A_conf"), "a")
    fig, axs = plt.subplots(1, 4, figsize=(FULL_W, 1.75))
    lams = sorted(RL)
    cols = seq(len(lams))
    G5 = amplification_per_coupling(0.5)
    Nc_lam, Lc_all = [], []
    for lam, c in zip(lams, cols):
        Ns, pc, n = curve(RL[lam])
        axs[0].plot(Ns, pc, "o-", color=c, label=f"{lam:g}")
        axs[1].plot(lam**2 * Ns * G5, pc, "o-", color=c, ms=2.5, lw=0.9)
        Nc, _ = fit_Nc(Ns, pc, n)
        Nc_lam.append(Nc)
        Lc_all.append(lam**2 * Nc * G5)
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"number of problems $N$")
    axs[0].set_ylabel("P(collapse)")
    axs[0].legend(title=r"$\lambda$", loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, fontsize=5.2,
                  title_fontsize=5.5, handlelength=0.9, columnspacing=0.6, borderaxespad=0.1)
    panel_label(axs[0], "a")

    acols = [CAT[1]] * len(RA)
    Nc_a = []
    for a in sorted(RA):
        Ns, pc, n = curve(RA[a])
        G = amplification_per_coupling(a)
        axs[1].plot(0.01 * Ns * G, pc, "s--", color=CAT[1], ms=2.5, lw=0.9, alpha=0.8)
        if len(Ns) > 2:
            Nc, _ = fit_Nc(Ns, pc, n)
            Nc_a.append((a, Nc))
            Lc_all.append(0.01 * Nc * G)
    axs[1].plot([], [], "o-", color=cols[-2], label=r"vary $\lambda$")
    axs[1].plot([], [], "s--", color=CAT[1], label=r"vary $a$")
    axs[1].set_xscale("log")
    axs[1].set_xlabel(r"$\Lambda=\lambda^2 N \ln[F^*/F(a)]$")
    axs[1].legend(loc="lower right", fontsize=5.5, handlelength=1.6, borderaxespad=0.2)
    panel_label(axs[1], "b")
    Lc = float(np.exp(np.mean(np.log(Lc_all))))
    axs[1].axvline(Lc, color=MUTED, lw=0.8, ls=":")

    lams_a = np.array(lams)
    Nc_lam = np.array(Nc_lam)
    slope = np.polyfit(np.log(lams_a), np.log(Nc_lam), 1)[0]
    ax = axs[2]
    ax.plot(lams_a, Nc_lam, "o", color=CAT[0], label=f"sim. (slope {slope:.2f})")
    lg = np.geomspace(0.03, 0.3, 50)
    ax.plot(lg, Lc / (lg**2 * G5), color=INK2, lw=0.9, ls="--", label=r"$\Lambda_c/(\lambda^2 G)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"coupling $\lambda$")
    ax.set_ylabel(r"$N_c$")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.2, borderaxespad=0.2)
    ax.set_ylim(30, 4e4)
    panel_label(ax, "c")

    ax = axs[3]
    aa = np.array([x[0] for x in Nc_a])
    nn = np.array([x[1] for x in Nc_a])
    ag = np.linspace(0.05, critical_logit() - 0.02, 200)
    ax.axvspan(critical_logit(), 1.8, color=SHADE, lw=0)
    ax.plot(ag, Lc / (0.01 * amplification_per_coupling(ag)), color=INK2, lw=0.9, ls="--", label="theory")
    ax.plot(aa, nn, "s", color=CAT[1], label="simulation")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.8)
    ax.set_ylim(80, 2e5)
    ax.text(1.56, 300, "stable\nfor all $N$", fontsize=5.5, ha="center", color=INK2)
    ax.set_xlabel(r"initial confidence $a=|z_i(0)|$")
    ax.set_ylabel(r"$N_c$")
    ax.legend(loc="upper left", fontsize=5.8)
    panel_label(ax, "d")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(FIGURES, "fig_scaling.pdf"))
    SUMMARY["Lambda_c"] = Lc
    SUMMARY["Lambda_c_values"] = Lc_all
    SUMMARY["Nc_vs_lambda"] = dict(zip(map(float, lams), map(float, Nc_lam)))
    SUMMARY["Nc_lambda_slope"] = float(slope)
    SUMMARY["Nc_vs_a"] = {float(a): float(n) for a, n in Nc_a}


if __name__ == "__main__":
    for name in sys.argv[1:] or ["mechanism", "scaling"]:
        globals()[f"fig_{name}"]()
    path = os.path.join(RESULTS, "summary_a.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(SUMMARY)
    json.dump(old, open(path, "w"), indent=1)
    print(json.dumps(SUMMARY, indent=1)[:1500])
