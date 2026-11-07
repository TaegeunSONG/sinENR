"""
Rebuild the benchmark figure (frequency resolution | denoising trace |
denoising error) without needing the original figdata.npz: the one quantity
that has to be recomputed is the denoising reconstruction, and it is
reproduced here from the same seeds as benchmark.py / benchmark_extra.py.

Usage:  python make_fig_benchmark.py   ->  fig3_benchmark.pdf / .png
                                           figdata_denoise.npz
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sinenr import fit_sinenr, predict, rmse, multi_tone

C_ENR, C_FFT, C_TRUE, META_GREY = "#0072B2", "#D55E00", "#444444", "#888888"
BASE, SECOND, TICK = 8, 7, 6
F_C, N_C, X_MAX_C, SIGMA_C = (5.0, 13.3, 21.7, 30.0), 512, 1.0, 1.0
PHI_C = np.random.default_rng(0).uniform(-np.pi, np.pi, len(F_C))   # as in benchmark.py


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


def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=BASE + 1, va="bottom", ha="left")


def main():
    apply_style()
    here = os.path.dirname(os.path.abspath(__file__))
    cache = os.path.join(here, "figdata_denoise.npz")

    if os.path.exists(cache):
        Z = np.load(cache)
        xC, cleanC, noisyC, predC = Z["xC"], Z["cleanC"], Z["noisyC"], Z["predC"]
    else:
        xC, cleanC = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
        noisyC = cleanC + np.random.default_rng(20_001).normal(0, SIGMA_C, N_C)
        L = float(np.ptp(xC))
        f_ = fit_sinenr(xC, noisyC, seed=1, n_units=100,
                        omega_max=2 * np.pi * 40.0 * L, lam_l1=0.1,
                        epochs=6000, lr=1e-2, n_restarts=3)
        predC = predict(f_, xC)
        np.savez(cache, xC=xC, cleanC=cleanC, noisyC=noisyC, predC=predC)
        print(f"  refit denoising: RMSE vs clean {rmse(predC, cleanC):.4f}", flush=True)

    raw = pd.read_csv(os.path.join(here, "results", "results_raw.csv"))
    A, Cc = raw[raw.task == "A"], raw[raw.task == "C"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    ax = axes[0]
    meths = ["SinENR", "FFT (8x zero-pad)", "FFT (peak-pick)"]
    cols = {"SinENR": C_ENR, "FFT (8x zero-pad)": C_FFT, "FFT (peak-pick)": "#E8A87C"}
    for i, m in enumerate(meths):
        v = A[A.method == m].f_err.values
        if m == "SinENR":
            ax.scatter(np.full(len(v), i) + (np.random.default_rng(1).random(len(v)) - .5) * .22,
                       v, s=9, color=cols[m], alpha=.65, lw=0)
            ax.plot([i - .24, i + .24], [np.median(v)] * 2, color="black", lw=1.6)
        else:
            ax.plot([i - .24, i + .24], [v[0]] * 2, color=cols[m], lw=2.2)
    ax.set_yscale("log"); ax.set_xticks(range(3))
    ax.set_xticklabels(["SinENR", "FFT\n8$\\times$ pad", "FFT\npeak"], fontsize=TICK)
    ax.set_ylabel("mean $|\\hat{f}-f|$  (cycles)")
    ax.set_xlim(-0.6, 2.6)
    panel_letter(ax, "a", dx=-0.40)

    ax = axes[1]
    sel = xC <= 0.25
    ax.plot(xC[sel], noisyC[sel], "o", ms=1.8, color="#C8C8C8", mew=0,
            label="noisy ($\\sigma$ = 1)")
    ax.plot(xC[sel], cleanC[sel], color=C_TRUE, lw=0.9, label="clean")
    ax.plot(xC[sel], predC[sel], color=C_ENR, lw=1.1, label="SinENR")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.margins(0.04)
    ax.legend(loc="lower center", ncol=3, handlelength=1.1, fontsize=TICK,
              columnspacing=0.8)
    panel_letter(ax, "b", dx=-0.38)

    ax = axes[2]
    order_c = ["No denoising", "FFT hard-threshold", "SinENR"]
    ccol = {"No denoising": "#999999", "FFT hard-threshold": C_FFT, "SinENR": C_ENR}
    for i, m in enumerate(order_c):
        v = Cc[Cc.method == m].denoise_rmse.values
        ax.scatter(np.full(len(v), i) + (np.random.default_rng(2).random(len(v)) - .5) * .22,
                   v, s=9, color=ccol[m], alpha=.65, lw=0)
        ax.plot([i - .24, i + .24], [np.median(v)] * 2, color="black", lw=1.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["none", "FFT\ntop-4", "SinENR"], fontsize=TICK)
    ax.set_ylabel("RMSE vs clean signal")
    ax.set_xlim(-0.6, 2.6); ax.margins(y=0.12)
    panel_letter(ax, "c", dx=-0.40)

    fig.subplots_adjust(wspace=0.46)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(here, f"fig3_benchmark.{ext}"), dpi=300,
                    bbox_inches="tight")
    print("wrote fig3_benchmark.pdf / .png")


if __name__ == "__main__":
    main()
