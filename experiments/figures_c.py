"""Figure and summary for Stage C (small Transformer)."""
import json
import os

import numpy as np

from common import FIGURES, RESULTS, load
from plotstyle import CAT, COL_W, FULL_W, INK2, MUTED, panel_label, plt, setup

setup()
COLLAPSE = 0.5


def key(d):
    return (d["arch"], d["base"], d["tau"], d["rho"])


def scores(runs, cells, n_boot=5000):
    """Score the predictors against trivial baselines: run-level Brier score and AUC,
    cell-level sign agreement, and a bootstrap over conditions (cells of a condition are correlated)."""
    by = {(c["cond"], c["N"]): c for c in cells}
    y = np.array([float(r["final"]["phi"] > COLLAPSE) for r in runs])
    cr = [by[(key(r), r["N"])] for r in runs]
    rate = float(y.mean())
    preds = {"surrogate": [c["p_sur"] for c in cr], "kernel_meanfield": [c["kmf"] for c in cr],
             "lambda_threshold": [float(c["Lambda"] > 2.64) for c in cr],
             "always_collapse": [1.0] * len(y), "always_stable": [0.0] * len(y), "base_rate": [rate] * len(y)}

    def auc(s):
        pos, neg = s[y == 1], s[y == 0]
        return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))

    out = dict(base_rate=rate, mean_surrogate=float(np.mean(preds["surrogate"])))
    for k, q in preds.items():
        q = np.array(q)
        out[k] = dict(brier=float(np.mean((q - y) ** 2)), run_agree=float(np.mean((q > 0.5) == (y > 0.5))))
    out["surrogate"]["auc"] = auc(np.array(preds["surrogate"]))
    po = np.array([c["p_obs"] for c in cells])
    ps = np.array([c["p_sur"] for c in cells])
    out["cell_agree"] = dict(surrogate=float(np.mean((ps > 0.5) == (po > 0.5))), always_collapse=float(np.mean(po > 0.5)))
    conds = sorted({c["cond"] for c in cells})
    idx = {c: [i for i, x in enumerate(cells) if x["cond"] == c] for c in conds}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[conds[j]] for j in rng.choice(len(conds), len(conds))])
        boot.append((np.mean((ps[ii] > 0.5) == (po[ii] > 0.5)), np.mean(po[ii] > 0.5)))
    boot = np.array(boot)
    out["cell_agree_ci95"] = dict(surrogate=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
                                  always_collapse=np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
                                  difference=np.percentile(boot[:, 0] - boot[:, 1], [2.5, 97.5]).tolist())
    return out


def main():
    runs = json.load(open(os.path.join(RESULTS, "C_runs.json")))
    kern = {key(r): r for r in json.load(open(os.path.join(RESULTS, "C_kernel.json")))}
    sur = load("C_surrogate")
    kmf = json.load(open(os.path.join(RESULTS, "C_kmf.json")))
    pred = {}
    for s, k in zip(sur, kmf):
        pred[(key(k), k["N"])] = dict(p_sur=s["p_collapse"], kmf=float(k["kmf_phi"] > COLLAPSE), Lambda=k["Lambda"])
    obs = {}
    for r in runs:
        obs.setdefault((key(r), r["N"]), []).append(r)
    cells = []
    for (c, N), rs in sorted(obs.items()):
        p = np.mean([x["final"]["phi"] > COLLAPSE for x in rs])
        cells.append(dict(cond=c, N=N, p_obs=float(p), acc=float(np.mean([x["final"]["acc"] for x in rs])),
                          test_acc=float(np.mean([x["final"]["test_acc"] for x in rs])),
                          reward=float(np.mean([x["final"]["reward"] for x in rs])), **pred[(c, N)]))

    fig, axs = plt.subplots(1, 4, figsize=(FULL_W, 1.9))
    # (a) training curves
    ax = axs[0]
    examples = [(("plain", 90, 2.0, 0.0), 512, CAT[1]), (("plain", 120, 1.0, 0.0), 512, CAT[0])]
    for c, N, col in examples:
        r = [x for x in obs[(c, N)] if x["seed"] == 0][0]
        ep = np.array(r["traj"]["epoch"])
        ax.plot(ep, r["traj"]["reward"], color=col, ls="-")
        ax.plot(ep, r["traj"]["test_acc"], color=col, ls="--")
        ax.text(ep[-1] * 1.02, r["traj"]["test_acc"][-1], f"base{c[1]}, $\\tau$={c[2]:g}", color=col, fontsize=5.5, va="center")
    ax.plot([], [], color=INK2, ls="-", label="MV reward")
    ax.plot([], [], color=INK2, ls="--", label="held-out acc.")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("epoch")
    ax.legend(loc="lower left", fontsize=5.5)
    panel_label(ax, "a")

    # (b) plain network: observed collapse across N
    ax = axs[1]
    plain = sorted({c["cond"] for c in cells if c["cond"][0] == "plain" and c["cond"][1] == 90})
    for j, c in enumerate(plain):
        cs = sorted([x for x in cells if x["cond"] == c], key=lambda x: x["N"])
        Ns = [x["N"] for x in cs]
        ax.plot(Ns, [x["p_obs"] for x in cs], "o-", color=CAT[j], ms=2.5, lw=1.0, label=f"$\\tau$={c[2]:g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$N$ (plain, base 90)")
    ax.set_ylabel("P(collapse)")
    ax.set_ylim(-0.03, 1.08)
    ax.legend(fontsize=5.5, loc="lower right", handlelength=1.4)
    panel_label(ax, "b")

    # (c) antisymmetric network: protection and its removal by rho
    ax = axs[2]
    anti = sorted({c["cond"][:3] for c in cells if c["cond"][0] == "antisym"})
    for j, a in enumerate(anti):
        rhos = sorted({c["cond"][3] for c in cells if c["cond"][:3] == a})
        po = [np.mean([x["p_obs"] for x in cells if x["cond"] == a + (r,)]) for r in rhos]
        ps = [np.mean([x["p_sur"] for x in cells if x["cond"] == a + (r,)]) for r in rhos]
        xs = np.array(rhos) + 0.01
        ax.plot(xs, po, "o-", color=CAT[j], ms=2.5, lw=1.0, label=f"{a[1]}, {a[2]:g}")
        ax.plot(xs, ps, ls=":", color=CAT[j], lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.3, 1, 3])
    ax.set_xticklabels(["0", "0.3", "1", "3"])
    ax.minorticks_off()
    ax.set_xlabel(r"added coupling $\rho_b=\lambda_b^2/k_d$")
    ax.set_ylabel("P(collapse), avg. over $N$")
    ax.legend(title=r"base, $\tau$", fontsize=5, title_fontsize=5.5, loc="upper left", handlelength=1.2)
    panel_label(ax, "c")

    # (d) calibration of the parameter-free surrogate
    ax = axs[3]
    po = np.array([x["p_obs"] for x in cells])
    ps = np.array([x["p_sur"] for x in cells])
    isplain = np.array([x["cond"][0] == "plain" for x in cells])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    ax.scatter(ps[~isplain], po[~isplain], s=9, color=CAT[0], alpha=0.75, label="antisym", lw=0)
    ax.scatter(ps[isplain], po[isplain], s=9, color=CAT[1], alpha=0.75, label="plain", lw=0, marker="s")
    r = np.corrcoef(ps, po)[0, 1]
    agree = np.mean((ps > 0.5) == (po > 0.5))
    ax.set_title(f"$r$={r:.2f}, sign agreement {100 * agree:.0f}%", color=INK2, fontsize=6.5)
    ax.set_xlabel("predicted P(collapse)")
    ax.set_ylabel("observed P(collapse)")
    ax.legend(loc="lower right", fontsize=5.5)
    panel_label(ax, "d")
    fig.tight_layout(w_pad=0.7)
    fig.savefig(os.path.join(FIGURES, "fig_nn.pdf"))

    # summary numbers
    kmf_agree = np.mean([(x["kmf"] > 0.5) == (x["p_obs"] > 0.5) for x in cells])
    lam_agree = np.mean([(x["Lambda"] > 2.64) == (x["p_obs"] > 0.5) for x in cells])
    summ = dict(n_cells=len(cells), n_runs=len(runs), surrogate_r=float(r), surrogate_agree=float(agree),
                kmf_agree=float(kmf_agree), lambda_threshold_agree=float(lam_agree),
                mean_abs_err=float(np.mean(np.abs(ps - po))), scores=scores(runs, cells),
                cells=[{**x, "cond": list(x["cond"])} for x in cells],
                kernel=[{**v, "key": list(k)} for k, v in kern.items()])
    if os.path.exists(os.path.join(RESULTS, "C_remedy.json")):
        rem = json.load(open(os.path.join(RESULTS, "C_remedy.json")))
        tab = {}
        for x in rem:
            tab.setdefault((x["beta"], x["N"]), []).append(x)
        summ["remedy"] = [dict(beta=b, N=N, p_collapse=float(np.mean([x["final"]["phi"] > COLLAPSE for x in v])),
                               acc=float(np.mean([x["final"]["acc"] for x in v])),
                               test_acc=float(np.mean([x["final"]["test_acc"] for x in v])),
                               conf=float(np.mean([x["final"]["conf"] for x in v])))
                          for (b, N), v in sorted(tab.items())]
    json.dump(summ, open(os.path.join(RESULTS, "summary_c.json"), "w"), indent=1)
    print({k: v for k, v in summ.items() if k not in ("cells", "kernel")})


def fig_remedy():
    """Marginal-entropy bonus on the plain base-90 network at tau=2: order parameter and accuracy vs beta."""
    rem = json.load(open(os.path.join(RESULTS, "C_remedy.json")))
    kern = {key(r): r for r in json.load(open(os.path.join(RESULTS, "C_kernel.json")))}
    acc0 = kern[("plain", 90, 2.0, 0.0)]["acc0"]
    betas = sorted({x["beta"] for x in rem})
    Ns = sorted({x["N"] for x in rem})
    xpos = {b: i for i, b in enumerate(betas)}
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 1.45))
    for ax, field, ylab in [(axs[0], "phi", r"final $\Phi$"), (axs[1], "test_acc", "held-out accuracy")]:
        for j, N in enumerate(Ns):
            off = (j - 0.5) * 0.12
            for b in betas:
                ys = [x["final"][field] for x in rem if x["beta"] == b and x["N"] == N]
                ax.scatter(np.full(len(ys), xpos[b] + off), ys, s=5, color=CAT[j], alpha=0.35, lw=0)
            mean = [np.mean([x["final"][field] for x in rem if x["beta"] == b and x["N"] == N]) for b in betas]
            ax.plot([xpos[b] + off for b in betas], mean, "o-", color=CAT[j], ms=2.5, lw=1.0, label=f"$N$={N}")
        ax.axvline(xpos[0.5], color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_xlabel(r"bonus weight $\beta$")
        ax.set_ylabel(ylab)
    axs[0].axhline(COLLAPSE, color=MUTED, lw=0.8, ls=":")
    axs[0].text(len(betas) - 0.6, COLLAPSE - 0.03, "collapse threshold", fontsize=5.5, color=INK2, ha="right", va="top")
    axs[0].text(xpos[0.5] + 0.08, 1.0, r"$\beta=\frac{1}{2}$", fontsize=5.5, color=INK2, va="top")
    axs[0].set_ylim(-0.03, 1.05)
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), fontsize=5.5, handlelength=1.2, ncol=2, columnspacing=0.8)
    axs[1].axhline(acc0, color=MUTED, lw=0.8, ls=":")
    axs[1].text(1.5, acc0 + 0.008, "initial accuracy", fontsize=5.5, color=INK2, ha="center", va="bottom")
    axs[1].axhline(0.5, color=MUTED, lw=0.8, ls=":")
    axs[1].text(0.5, 0.492, "chance", fontsize=5.5, color=INK2, ha="center", va="top")
    axs[1].set_ylim(0.45, 0.9)
    panel_label(axs[0], "a")
    panel_label(axs[1], "b")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIGURES, "fig_nn_remedy.pdf"))


if __name__ == "__main__":
    main()
    fig_remedy()
