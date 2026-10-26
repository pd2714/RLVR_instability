"""Matplotlib style for paper figures (single-column ICML width ~3.25in)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots, fixed order (identity)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# sequential blue ramp (ordinal magnitude: light -> dark), lightest usable step first
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SHADE = "#f3dccf"   # unstable-window shading (tint of slot 2)

COL_W = 3.25
FULL_W = 6.75


def seq(n):
    """n ordinal colors from the blue ramp, evenly spaced, light -> dark."""
    if n == 1:
        return [SEQ[5]]
    return [SEQ[round(i * (len(SEQ) - 1) / (n - 1))] for i in range(n)]


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right", color=INK)
