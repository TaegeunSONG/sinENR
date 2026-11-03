"""
Figure for the islet close-pair robustness checks (Referee 1, point 4).

Four panels, all built from the CSV tables written by islet_robustness.py:
  (a) periodogram of a trajectory 8x longer than the analysed record, over the
      dominant-cluster window, with the retained components of the analysed
      record and its DFT grid spacing drawn for scale;
  (b) the natural frequencies of the 600 oscillators that generated the trace,
      against those same retained components;
  (c) retained cluster components as the record length grows;
  (d) bootstrap distribution of the two components reported as a resolved
      pair, against their claimed separation.

Usage:  python make_fig_islet_robustness.py  ->  fig6_islet_robustness.pdf/.png
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

C_ENR, C_FFT, C_SPL, C_TRUE, META_GREY = ("#0072B2", "#D55E00", "#009E73",
                                          "#444444", "#888888")
BASE, SECOND, TICK = 8, 7, 6
PAIR = (0.180, 0.191)
CLUSTER = (0.12, 0.24)
REF_COND = "lam=0.001"          # the configuration of Fig. 4(f)


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": BASE,
        "axes.labelsize": BASE, "axes.titlesize": BASE, "legend.fontsize": SECOND,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.22, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=BASE + 1, va="bottom", ha="left")


def main():
    apply_style()
    spec = pd.read_csv(os.path.join(RES, "islet_longrun_spectrum.csv"))
    nat = pd.read_csv(os.path.join(RES, "islet_natural_frequencies.csv"))
    comp = pd.read_csv(os.path.join(RES, "islet_cluster_components.csv"))
    boot = pd.read_csv(os.path.join(RES, "islet_bootstrap_cluster.csv"))
    summ = pd.read_csv(os.path.join(RES, "islet_robustness_summary.csv"))

    def sval(check, quantity):
        return float(summ[(summ.check == check)
                          & (summ.quantity == quantity)].value.iloc[0])

    bin_short = sval("long trajectory", "dft bin analysed")
    ref = comp[(comp.check == "lambda_1") & (comp.condition == REF_COND)
               & comp.in_cluster].sort_values("f")

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.45))

    # ---- (a) long-trajectory periodogram -----------------------------
    ax = axes[0]
    ax.plot(spec.f, spec.amplitude, color=META_GREY, lw=0.9,
            label="long record, $L=600$")
    sc = ref.amplitude / ref.amplitude.max() * spec.amplitude.max()
    ax.vlines(ref.f, 0, sc, color=C_ENR, lw=1.1, alpha=0.9,
              label="SinENR components\n(analysed record)")
    for f0 in PAIR:
        ax.axvline(f0, color=C_FFT, ls=":", lw=0.8)
    ymax = spec.amplitude.max() * 1.30
    yb = ymax * 0.47
    ax.annotate("", xy=(0.2185 + bin_short / 2, yb), xytext=(0.2185 - bin_short / 2, yb),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=C_FFT))
    ax.text(0.2185, yb + ymax * 0.035, "$1/L$ of the\nanalysed record",
            fontsize=TICK, ha="center", va="bottom", color=C_FFT)
    ax.set_xlim(*CLUSTER)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("frequency  (cycles/time)")
    ax.set_ylabel("amplitude")
    ax.set_title("Long record shows a band,\nnot two lines")
    ax.legend(loc="upper left", fontsize=TICK - 0.5, handlelength=1.2)
    panel_letter(ax, "a")

    # ---- (b) natural frequencies of the generating oscillators -------
    ax = axes[1]
    ax.hist(nat.f_natural, bins=40, range=CLUSTER, color=C_SPL, alpha=0.55,
            edgecolor="none", label=r"$\omega_i/2\pi$ of the 600 oscillators")
    cnt, _ = np.histogram(nat.f_natural, bins=40, range=CLUSTER)
    stem = ref.amplitude / ref.amplitude.max() * cnt.max() * 0.78
    ax.vlines(ref.f, 0, stem, color=C_ENR, lw=1.1,
              label="SinENR components\n(height $\\propto$ amplitude)")
    ax.set_xlim(*CLUSTER)
    ax.set_ylim(0, cnt.max() * 1.32)
    ax.set_xlabel("frequency  (cycles/time)")
    ax.set_ylabel("oscillator count")
    for f0 in PAIR:
        ax.axvline(f0, color=C_FFT, ls=":", lw=0.8)
    ax.set_title("Cluster spans the natural-\nfrequency distribution")
    ax.legend(loc="upper left", fontsize=TICK - 0.5, handlelength=1.2)
    panel_letter(ax, "b")

    # ---- (c) record length -------------------------------------------
    ax = axes[2]
    rl = comp[(comp.check == "record length") & comp.in_cluster].copy()
    rl["N"] = rl.condition.str.replace("N=", "", regex=False).astype(int)
    Ns = sorted(rl.N.unique())
    pos = {n: i for i, n in enumerate(Ns)}
    for n, grp in rl.groupby("N"):
        ax.scatter([pos[n]] * len(grp), grp.f,
                   s=14 + 55 * grp.amplitude / rl.amplitude.max(),
                   color=C_ENR, alpha=0.75, edgecolors="none")
    cen = rl.groupby("N").apply(
        lambda g: np.average(g.f, weights=g.amplitude), include_groups=False)
    ax.plot([pos[n] for n in cen.index], cen.values, color=C_TRUE, lw=1.0,
            marker="o", ms=2.5, label="amplitude-weighted\ncentroid")
    for f0 in PAIR:
        ax.axhline(f0, color=C_FFT, ls=":", lw=0.8)
    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns], rotation=45, ha="right")
    ax.set_xlim(-0.5, len(Ns) - 0.5)
    ax.set_xlabel("record length  $N$  (samples)")
    ax.set_ylabel("frequency  (cycles/time)")
    ax.set_ylim(*CLUSTER)
    ax.set_title("Individual lines move,\nthe centroid does not")
    ax.legend(loc="upper right", fontsize=TICK - 0.5, handlelength=1.2)
    panel_letter(ax, "c")

    # ---- (d) bootstrap -----------------------------------------------
    ax = axes[3]
    bins = np.linspace(0.155, 0.225, 26)
    ax.hist(boot.f_near_lo, bins=bins, color=C_ENR, alpha=0.6, edgecolor="none",
            label="nearest to $0.180$")
    ax.hist(boot.f_near_hi, bins=bins, color=C_FFT, alpha=0.5, edgecolor="none",
            label="nearest to $0.191$")
    for f0 in PAIR:
        ax.axvline(f0, color=C_TRUE, ls=":", lw=0.8)
    ax.set_xlabel("frequency  (cycles/time)")
    ax.set_ylabel("bootstrap refits")
    sd_lo, sd_hi = boot.f_near_lo.std(), boot.f_near_hi.std()
    ax.text(0.02, 0.98,
            f"$\\Delta f$ claimed $=0.0104$\n"
            f"s.d. $={sd_lo:.4f}$, ${sd_hi:.4f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=TICK)
    ax.set_title("Scatter is comparable\nto the claimed gap")
    ax.legend(loc="upper right", fontsize=TICK - 0.5, handlelength=1.2)
    panel_letter(ax, "d")

    fig.subplots_adjust(wspace=0.34)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig6_islet_robustness.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("wrote fig6_islet_robustness.pdf / .png")
    print(f"  reference components in cluster: {np.round(ref.f.values, 4)}")
    print(f"  bootstrap sd near pair: {sd_lo:.5f}, {sd_hi:.5f}")


if __name__ == "__main__":
    main()
