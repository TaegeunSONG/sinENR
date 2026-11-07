"""
Regenerate every manuscript figure as vector PDF (IOP's preferred format) and
PNG, from the saved result files -- no refitting required.

Inputs : figdata.npz (make_figdata.py), alias_landscape.npz
         (alias_analysis.py), and the per-seed tables in results/ --
         results_raw.csv, results_B2_contiguous.csv, results_C2_lambda.csv,
         sparsity_tradeoff.csv, results_alias.csv
Outputs: figN_*.pdf and .png beside this script, which is where the
         manuscript's \includegraphics looks for them.

Usage:  python make_figdata.py && python alias_analysis.py && python make_figures.py
"""
from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE


def _csv(name):
    """Result tables live in results/; fall back to the script directory."""
    for cand in (os.path.join(HERE, "results", name), os.path.join(HERE, name)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"{name} not found in results/ or {HERE}. Run benchmark.py, "
        f"benchmark_extra.py and alias_analysis.py first.")


def _npz(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        maker = "make_figdata.py" if name == "figdata.npz" else "alias_analysis.py"
        raise FileNotFoundError(f"{name} not found; run {maker} first.")
    return path

C_ENR, C_FFT, C_SPL, C_TRUE, META_GREY = "#0072B2", "#D55E00", "#009E73", "#444444", "#888888"
BASE, SECOND, TICK = 8, 7, 6


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": BASE,
        "axes.labelsize": BASE, "axes.titlesize": BASE, "legend.fontsize": SECOND,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.18, dy=1.02):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=BASE + 1, va="bottom", ha="left")


def strip(ax, labels, values, color, jitter=0.12, seed=0):
    rng = np.random.default_rng(seed)
    for i, ys in enumerate(values):
        ys = np.asarray(ys)
        jit = (rng.random(ys.size) - 0.5) * 2 * jitter
        ax.scatter(np.full(ys.size, i) + jit, ys, s=10, color=color, alpha=0.6,
                   linewidths=0, zorder=2)
        ax.plot([i - 0.22, i + 0.22], [np.median(ys)] * 2, color="black", lw=1.6, zorder=3)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT}/{stem}.pdf (+ .png)")


# -------------------------------------------------------------------------
D = np.load(_npz("figdata.npz"))
LS = np.load(_npz("alias_landscape.npz"))
raw = pd.read_csv(_csv("results_raw.csv"))
b2 = pd.read_csv(_csv("results_B2_contiguous.csv"))
c2 = pd.read_csv(_csv("results_C2_lambda.csv"))
tr = pd.read_csv(_csv("sparsity_tradeoff.csv"))
al = pd.read_csv(_csv("results_alias.csv"))
A = raw[raw.task == "A"]; B = raw[raw.task == "B"]; Cc = raw[raw.task == "C"]
fft_err = float(A[A.method == "FFT (8x zero-pad)"].f_err.iloc[0]) * 1e3
fft_c = float(Cc[Cc.method == "FFT hard-threshold"].denoise_rmse.mean())
apply_style()


# ---- figure 1: model properties -----------------------------------------
def fig_properties():
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = GridSpec(2, 3, figure=fig, hspace=0.95, wspace=0.60, width_ratios=[1.35, 1, 1])

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(D["xd"], D["yd"], color=C_TRUE, lw=1.0, label="true signal")
    ax.plot(D["xA"], D["yA"], "o", ms=2.6, color=C_TRUE, mfc="white", mew=0.7,
            label="samples (N = 100)")
    ax.plot(D["xd"], D["pred_A"], color=C_ENR, lw=1.2, label="SinENR")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.margins(0.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3, handlelength=1.4)
    panel_letter(ax, "a", dx=-0.10)

    ax = fig.add_subplot(gs[0, 2])
    ax.semilogy(np.arange(1, len(D["hist_A"]) + 1), D["hist_A"], color=C_ENR, lw=0.9)
    ax.set_xlabel("epoch"); ax.set_ylabel("training MSE")
    ax.set_xticks([0, 1500, 3000]); ax.margins(x=0.04)
    panel_letter(ax, "b", dx=-0.44)

    ax = fig.add_subplot(gs[1, 0])
    ca, ck = D["comp_all"], D["comp_kept"]
    ax.plot(ca[:, 0], ca[:, 2], "o", ms=2.6, color="#BBBBBB", mew=0, label="pruned units")
    ax.plot(ck[:, 0], ck[:, 2], "o", ms=5, color=C_ENR, mew=0, label="retained")
    for ft in (0.7, 1.5):
        ax.axvline(ft, color=C_TRUE, ls=":", lw=0.8, zorder=0)
    ax.text(1.5, 1.12, "true $f$", color=C_TRUE, fontsize=TICK, ha="center")
    ax.set_xlabel("frequency  $|W_1^j|/2\\pi L$"); ax.set_ylabel("amplitude  $|W_2^j|$")
    ax.set_ylim(-0.06, 1.28); ax.margins(x=0.06)
    ax.legend(loc="center right", handlelength=1.0, handletextpad=0.3)
    panel_letter(ax, "c", dx=-0.32)

    ax = fig.add_subplot(gs[1, 1])
    # profile of the minimal-model slice along the frequency axis, with the
    # amplitude minimized out: this is where the local-minimum structure of
    # finite-record sinusoidal fitting is visible.
    w1 = D["W1g"][0, :]
    prof = D["Z"].min(axis=0)
    ax.semilogy(w1, prof, color=C_ENR, lw=0.9)
    loc = [i for i in range(1, len(prof) - 1)
           if prof[i] < prof[i - 1] and prof[i] < prof[i + 1]]
    ax.plot(w1[loc], prof[loc], "v", ms=2.6, color=C_TRUE, mew=0, clip_on=False)
    ax.set_xlabel("$W_1^1$")
    ax.set_ylabel("MSE (min over $W_2^1$)")
    ax.margins(x=0.03)
    panel_letter(ax, "d", dx=-0.52)

    ax = fig.add_subplot(gs[1, 2])
    dr, rc = D["draws"], D["ref_comp"]
    truth = [1.5, 0.7] if rc[0, 0] > rc[1, 0] else [0.7, 1.5]
    pairs = []
    for k, ft in enumerate(truth):
        v = dr[:, k, 0]; v = v[np.isfinite(v)]
        pairs.append((ft, (v - ft) * 1e3))
    pairs.sort(key=lambda p: p[0])
    strip(ax, [f"$f$ = {p[0]}" for p in pairs], [p[1] for p in pairs], C_ENR, jitter=0.10)
    ax.axhline(fft_err, color=C_FFT, ls="--", lw=0.9)
    ax.text(0.5, fft_err - 1.0, "8$\\times$ zero-padded\nFFT error", color=C_FFT,
            fontsize=TICK, ha="center", va="top")
    ax.axhline(0, color=C_TRUE, lw=0.6, zorder=0)
    ax.set_ylabel("$\\hat{f}-f$   ($10^{-3}$ cycles)")
    ax.set_ylim(-5, 17); ax.set_xlim(-0.5, 1.5)
    panel_letter(ax, "e", dx=-0.50)
    save(fig, "fig1_properties")


# ---- figure 2: qualitative recovery -------------------------------------
def fig_recovery():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    midx = D["midx"]; obs = D["obs"]
    g0, g1 = D["xA"][midx[0]], D["xA"][midx[-1]]
    ax.axvspan(g0, g1, color="#F0F0F0", zorder=0)
    ax.text((g0 + g1) / 2, 2.05, "masked (30 pts)", fontsize=TICK, ha="center", color="#666666")
    ax.plot(D["xd"], D["yd"], color=C_TRUE, lw=0.9, label="true signal")
    ax.plot(D["xA"][obs], D["yA"][obs], "o", ms=2.2, color=C_TRUE, mfc="white",
            mew=0.6, label="observed")
    ax.plot(D["xd"], D["splineB"], color=C_SPL, lw=1.1, ls="--", label="cubic spline")
    ax.plot(D["xd"], D["predB"], color=C_ENR, lw=1.2, label="SinENR")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_ylim(-3.4, 2.6); ax.margins(x=0.02)
    ax.legend(loc="lower center", ncol=2, handlelength=1.4, fontsize=TICK)
    panel_letter(ax, "a", dx=-0.16)

    ax = axes[1]
    sel = D["xC"] <= 0.25
    ax.plot(D["xC"][sel], D["noisyC"][sel], "o", ms=1.8, color="#C8C8C8", mew=0,
            label="noisy ($\\sigma$ = 1)")
    ax.plot(D["xC"][sel], D["cleanC"][sel], color=C_TRUE, lw=0.9, label="clean signal")
    ax.plot(D["xC"][sel], D["predC"][sel], color=C_ENR, lw=1.1, label="SinENR")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.margins(0.04)
    ax.legend(loc="lower center", ncol=3, handlelength=1.3, fontsize=TICK)
    panel_letter(ax, "b", dx=-0.16)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig2_recovery")


# ---- figure 3: benchmark ------------------------------------------------
def fig_benchmark():
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
    sel = D["xC"] <= 0.25
    ax.plot(D["xC"][sel], D["noisyC"][sel], "o", ms=1.8, color="#C8C8C8", mew=0,
            label="noisy ($\\sigma$ = 1)")
    ax.plot(D["xC"][sel], D["cleanC"][sel], color=C_TRUE, lw=0.9, label="clean")
    ax.plot(D["xC"][sel], D["predC"][sel], color=C_ENR, lw=1.1, label="SinENR")
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
    save(fig, "fig3_benchmark")


# ---- figure 4: sparsity / fidelity --------------------------------------
def fig_sparsity():
    g = tr.groupby("lam").agg(nc=("n_comp", "mean"), rm=("recon_rmse", "mean")).reset_index()
    gc = c2.groupby("lam").agg(nc=("n_comp", "mean"), rm=("denoise_rmse", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.plot(g.nc, g.rm, "-o", color=C_ENR, ms=4, lw=1.0)
    for _, r_ in g.iterrows():
        ax.annotate(f"{r_.lam:g}", (r_.nc, r_.rm), textcoords="offset points",
                    xytext=(4, 4), fontsize=TICK, color=META_GREY)
    ax.axvline(2, color=C_TRUE, ls=":", lw=0.9)
    ax.text(2.15, 0.0145, "true model\norder K = 2", fontsize=TICK, color=C_TRUE)
    ax.set_yscale("log"); ax.set_xlabel("components retained")
    ax.set_ylabel("reconstruction RMSE")
    panel_letter(ax, "a", dx=-0.17)

    ax = axes[1]
    ax.plot(gc.nc, gc.rm, "-o", color=C_ENR, ms=4, lw=1.0)
    for _, r_ in gc.iterrows():
        ax.annotate(f"{r_.lam:g}", (r_.nc, r_.rm), textcoords="offset points",
                    xytext=(4, 5), fontsize=TICK, color=META_GREY)
    ax.axhline(fft_c, color=C_FFT, ls="--", lw=0.9)
    ax.text(14, fft_c + 0.012, "FFT top-4", color=C_FFT, fontsize=TICK)
    ax.axvline(4, color=C_TRUE, ls=":", lw=0.9)
    ax.text(4.5, 0.52, "true model\norder K = 4", fontsize=TICK, color=C_TRUE)
    ax.set_xlabel("components retained"); ax.set_ylabel("RMSE vs clean signal")
    ax.margins(0.08)
    panel_letter(ax, "b", dx=-0.17)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig4_sparsity")


# ---- figure 5: aliasing -------------------------------------------------
def fig_alias():
    f_s, f_nyq = float(LS["f_s"]), float(LS["f_nyq"])
    u = al[al.sampling == "uniform"]
    summ = u.groupby("f_max").agg(n_true=("n_true", "mean"), n_alias=("n_alias", "mean"),
                                  ins=("in_sample_mse", "mean"), dns=("dense_rmse", "mean"),
                                  br=("branches_per_tone", "first")).reset_index()
    summ["frac_true_obs"] = summ.n_true / (summ.n_true + summ.n_alias)
    summ["frac_true_pred"] = 1.0 / summ.br
    fsc = LS["f_scan"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.semilogy(fsc, LS["mse_uniform"], color=C_ENR, lw=0.8, label="uniform grid")
    ax.semilogy(fsc, LS["mse_jitter"], color=C_FFT, lw=0.8, alpha=0.85, label="jittered 40 %")
    ax.axvline(f_nyq, color=C_TRUE, ls=":", lw=0.9)
    ax.text(f_nyq + 1.8, 2.5e-6, "Nyquist", fontsize=TICK, color=C_TRUE, rotation=90, va="bottom")
    for fa in [0.7, 32.3, 33.7, 65.3]:
        ax.plot(fa, 1.6e-6, "v", ms=3, color=C_TRUE, clip_on=False)
    ax.set_xlabel("frequency of one atom, $f$  (cycles)")
    ax.set_ylabel("training MSE")
    ax.set_ylim(1e-6, 60)
    ax.legend(loc="upper right", fontsize=TICK, handlelength=1.2, borderpad=0.2)
    panel_letter(ax, "a", dx=-0.34)

    ax = axes[1]
    xpos = np.arange(len(summ))
    ax.bar(xpos - 0.17, summ.frac_true_obs, width=0.34, color=C_ENR, label="recovered true $f$")
    ax.bar(xpos + 0.17, 1 - summ.frac_true_obs, width=0.34, color=C_FFT, label="recovered an alias")
    ax.plot(xpos - 0.17, summ.frac_true_pred, "D", ms=4, color="black", zorder=5,
            label="$1/n_\\mathrm{branch}$ prediction")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v:.2g}" for v in summ.f_max / f_nyq], fontsize=TICK)
    ax.set_xlabel("frequency prior bound  $f_\\max / f_\\mathrm{Nyq}$")
    ax.set_ylabel("fraction of components")
    ax.set_ylim(0, 1.42)
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.1, borderpad=0.2)
    panel_letter(ax, "b", dx=-0.30)

    ax = axes[2]
    ax.plot(xpos, summ.ins, "-o", color="#999999", ms=4, lw=1.0, label="training MSE")
    ax.plot(xpos, summ.dns, "-o", color=C_ENR, ms=4, lw=1.0, label="dense-grid RMSE")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{v:.2g}" for v in summ.f_max / f_nyq], fontsize=TICK)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("frequency prior bound  $f_\\max / f_\\mathrm{Nyq}$")
    ax.set_ylabel("error"); ax.set_ylim(1e-5, 6)
    ax.legend(loc="center left", fontsize=TICK, handlelength=1.3, borderpad=0.2)
    panel_letter(ax, "c", dx=-0.34)
    fig.subplots_adjust(wspace=0.44)
    save(fig, "fig5_alias")


if __name__ == "__main__":
    fig_properties(); fig_benchmark(); fig_sparsity(); fig_alias()
    print(f"all figures written to {OUT}")
