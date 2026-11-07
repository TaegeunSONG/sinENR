"""
Rebuild the missing-data figure with the mask geometry made explicit.

The earlier version showed a single gap and one reconstruction, which does not
carry the result: the conclusion depends on *how* the missing samples are
distributed, and the interpretable readout survives a 30 % contiguous loss.
Four panels: (a) the three mask geometries, (b) reconstruction across the
single block, (c) error by geometry, (d) the spectral readout recovered from
the gapped fit.

Usage:  python make_fig_missing.py      ->  fig_missing.pdf / .png
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sinenr import (fit_sinenr, predict, spectral_readout, rmse, two_tone)

C_ENR, C_FFT, C_SPL, C_TRUE, META_GREY = "#0072B2", "#D55E00", "#009E73", "#444444", "#888888"
BASE, SECOND, TICK = 8, 7, 6
F_TRUE = (0.7, 1.5)
N, X_MAX, FRAC = 100, 3.0, 0.30


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


def masks(n_blocks, seed=1):
    """Reproduce the masks of benchmark_extra.task_B2 / benchmark.task_B."""
    n_mask = int(round(FRAC * N))
    if n_blocks == 0:                                    # isolated dropouts
        rng = np.random.default_rng(10_000 + seed)
        return np.sort(rng.choice(np.arange(1, N - 1), n_mask, replace=False))
    rng = np.random.default_rng(30_000 + 100 * n_blocks + seed)
    blk, starts, tries = n_mask // n_blocks, [], 0
    while len(starts) < n_blocks and tries < 1000:
        s = int(rng.integers(1, N - blk - 1))
        cand = set(range(s, s + blk))
        if all(not (cand & set(range(m, m + blk))) for m in starts):
            starts.append(s)
        tries += 1
    return np.sort(np.concatenate([np.arange(m, m + blk) for m in starts]))


def islet_case(here, n_blocks=1, frac=0.30, seed_mask=778, stride=4, cache=True):
    """Fit SinENR to the islet glucose trace with a contiguous block withheld."""
    import islet_model as IM
    from scipy.interpolate import CubicSpline as _CS
    cp = os.path.join(here, f"islet_cache_{n_blocks}.npz")
    if cache and os.path.exists(cp):
        Z = np.load(cp)
        return dict(t=Z["t"], g=Z["g"], mi=Z["mi"], pred=Z["pred"],
                    spl=Z["spl"], lin=Z["lin"], comp=Z["comp"], dense=Z["dense"],
                    pred_dense=Z["pred_dense"], spl_dense=Z["spl_dense"])
    t_, g_, _ = IM.simulate(seed=0, burn_in=0.25)
    t_, g_ = t_[::stride], g_[::stride]
    L = float(np.ptp(t_))
    n_mask = int(round(frac * len(t_))); blk = n_mask // n_blocks
    rng = np.random.default_rng(seed_mask); starts = []
    while len(starts) < n_blocks:
        s = int(rng.integers(1, len(t_) - blk - 1))
        if all(not (set(range(s, s + blk)) & set(range(m, m + blk))) for m in starts):
            starts.append(s)
    mi = np.sort(np.concatenate([np.arange(m, m + blk) for m in starts]))
    obs = np.setdiff1d(np.arange(len(t_)), mi)
    f_ = fit_sinenr(t_[obs], g_[obs], seed=1, n_units=100,
                    omega_max=2 * np.pi * 1.0 * L, lam_l1=1e-3,
                    epochs=3000, lr=1e-2, n_restarts=3)
    cs = _CS(t_[obs], g_[obs])
    dense = np.linspace(t_[0], t_[-1], 3000)
    out = dict(t=t_, g=g_, mi=mi,
               pred=predict(f_, t_[mi]), spl=cs(t_[mi]),
               lin=np.interp(t_[mi], t_[obs], g_[obs]),
               comp=spectral_readout(f_, tau_rel=0.05),
               dense=dense, pred_dense=predict(f_, dense), spl_dense=cs(dense))
    if cache:
        np.savez(cp, **out)
    print(f"  islet ({n_blocks} block): gap RMSE SinENR {rmse(out['pred'], g_[mi]):.4f}"
          f" | spline {rmse(out['spl'], g_[mi]):.4f}"
          f" | linear {rmse(out['lin'], g_[mi]):.4f}"
          f" | signal sd {g_.std():.4f} | n_comp {len(out['comp'])}", flush=True)
    return out


def main():
    apply_style()
    here = os.path.dirname(os.path.abspath(__file__))
    x, y = two_tone(N=N, x_max=X_MAX, f=F_TRUE)
    L = float(np.ptp(x))
    xd = np.linspace(0, X_MAX, 2000)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_TRUE)
    kw = dict(n_units=100, omega_max=2 * np.pi * 2.5 * L, lam_l1=5e-3,
              epochs=3000, lr=1e-2, n_restarts=3)

    geoms = [(0, "30 isolated points"), (3, "3 blocks of 10"), (1, "one block of 30")]
    M = {nb: masks(nb) for nb, _ in geoms}

    # fits needed for (b) and (d)
    fits = {}
    for nb in (1, 0):
        obs = np.setdiff1d(np.arange(N), M[nb])
        fits[nb] = fit_sinenr(x[obs], y[obs], seed=1, **kw)
        print(f"  fitted nb={nb}: gap RMSE "
              f"{rmse(predict(fits[nb], x[M[nb]]), y[M[nb]]):.5f}", flush=True)

    raw = pd.read_csv(os.path.join(here, "results", "results_raw.csv"))
    b2 = pd.read_csv(os.path.join(here, "results", "results_B2_contiguous.csv"))
    B = raw[raw.task == "B"]

    isl = islet_case(here, n_blocks=1)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.80, wspace=0.42)

    # ---- (a) the three mask geometries ------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for row, (nb, name) in enumerate(geoms):
        obs = np.setdiff1d(np.arange(N), M[nb])
        ax.vlines(x[M[nb]], -row - 0.30, -row + 0.30, color="#DCDCDC", lw=1.1)
        ax.vlines(x[obs], -row - 0.30, -row + 0.30, color=C_ENR, lw=1.1)
    ax.set_yticks([0, -1, -2])
    ax.set_yticklabels(["30 isolated\npoints", "3 blocks\nof 10", "one block\nof 30"],
                       fontsize=TICK)
    ax.set_ylim(-2.7, 0.7); ax.set_xlim(-0.05, X_MAX + 0.05)
    ax.set_xlabel("x")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.vlines([], [], [], color=C_ENR, lw=1.6, label="observed")
    ax.vlines([], [], [], color="#DCDCDC", lw=1.6, label="missing")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.10), ncol=2,
              handlelength=0.7, handletextpad=0.4, columnspacing=1.0,
              fontsize=TICK)
    panel_letter(ax, "a", dx=-0.22)

    # ---- (b) reconstruction across the single block -----------------
    ax = fig.add_subplot(gs[0, 1])
    mi = M[1]; obs = np.setdiff1d(np.arange(N), mi)
    g0, g1 = x[mi[0]], x[mi[-1]]
    cs = CubicSpline(x[obs], y[obs])
    ax.axvspan(g0, g1, color="#F2F2F2", zorder=0)
    ax.plot(xd, yd, color=C_TRUE, lw=0.9, label="true signal")
    ax.plot(x[obs], y[obs], "o", ms=2.0, color=C_TRUE, mfc="white", mew=0.6,
            label="observed")
    ax.plot(xd, cs(xd), color=C_SPL, lw=1.1, ls="--", label="cubic spline")
    ax.plot(xd, predict(fits[1], xd), color=C_ENR, lw=1.2, label="SinENR")
    lo, hi = max(0, g0 - 0.55), min(X_MAX, g1 + 0.55)
    ax.set_xlim(lo, hi); ax.set_ylim(-2.6, 2.3)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="lower center", ncol=2, handlelength=1.3, fontsize=TICK,
              columnspacing=1.0)
    panel_letter(ax, "b", dx=-0.17)

    # ---- (c) error by geometry --------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    series = {"SinENR": C_ENR, "Cubic spline": C_SPL, "Linear interp": "#999999"}
    data = {
        "30 isolated": {m: B[B.method == m].gap_rmse.values for m in series},
        "3 blocks": {m: b2[(b2.n_blocks == 3) & (b2.method == m)].gap_rmse.values for m in series},
        "one block": {m: b2[(b2.n_blocks == 1) & (b2.method == m)].gap_rmse.values for m in series},
    }
    for gi, (gname, dd) in enumerate(data.items()):
        for mi_, (m, v) in enumerate(dd.items()):
            off = (mi_ - 1) * 0.26
            ax.plot([gi + off - 0.1, gi + off + 0.1], [np.mean(v)] * 2,
                    color=series[m], lw=2.0, solid_capstyle="butt")
            ax.vlines(gi + off, np.mean(v) - np.std(v, ddof=1),
                      np.mean(v) + np.std(v, ddof=1), color=series[m], lw=0.9)
    ax.set_yscale("log"); ax.set_xticks(range(3))
    ax.set_xticklabels(data.keys(), fontsize=TICK)
    ax.set_ylabel("gap RMSE"); ax.set_xlim(-0.55, 2.55)
    for m, c in series.items():
        ax.plot([], [], color=c, lw=2.0, label=m)
    ax.legend(loc="lower right", fontsize=TICK, handlelength=1.1)
    panel_letter(ax, "c", dx=-0.17)

    # ---- (d) the islet trace with the block withheld -----------------
    ti, gi, mi_i = isl["t"], isl["g"], isl["mi"]
    obs_i = np.setdiff1d(np.arange(len(ti)), mi_i)
    t0, t1 = ti[mi_i[0]], ti[mi_i[-1]]
    ax = fig.add_subplot(gs[1, 0])
    ax.axvspan(t0, t1, color="#F2F2F2", zorder=0)
    ax.plot(ti, gi, color=C_TRUE, lw=0.7)
    ax.plot(ti[obs_i], gi[obs_i], "o", ms=0.6, color=C_ENR, mew=0)
    ax.text((t0 + t1) / 2, 0.63, "withheld", fontsize=TICK, ha="center", color="#666666")
    ax.set_xlabel("time"); ax.set_ylabel("$g-\\langle g\\rangle$")
    ax.set_ylim(-0.72, 0.78); ax.margins(x=0.02)
    panel_letter(ax, "d", dx=-0.26)

    # ---- (e) reconstruction across the physiological gap -------------
    ax = fig.add_subplot(gs[1, 1])
    ax.axvspan(t0, t1, color="#F2F2F2", zorder=0)
    ax.plot(isl["dense"], isl["spl_dense"], color=C_SPL, lw=1.0, ls="--",
            label="cubic spline")
    ax.plot(ti[mi_i], isl["lin"], color="#999999", lw=1.0, label="linear")
    ax.plot(ti, gi, color=C_TRUE, lw=0.8, label="true")
    ax.plot(isl["dense"], isl["pred_dense"], color=C_ENR, lw=1.1, label="SinENR")
    ax.set_xlim(t0 - 7, t1 + 7); ax.set_ylim(-0.78, 1.45)
    ax.set_xlabel("time"); ax.set_ylabel("$g-\\langle g\\rangle$")
    ax.legend(loc="upper center", ncol=4, fontsize=5.6, handlelength=0.9,
              columnspacing=0.6, handletextpad=0.35,
              bbox_to_anchor=(0.46, 1.055))
    panel_letter(ax, "e", dx=-0.26)

    # ---- (f) components recovered, against the Fourier bin width -----
    ax = fig.add_subplot(gs[1, 2])
    c = isl["comp"]; Li = float(np.ptp(ti)); dfbin = 1.0 / Li
    ax.vlines(c[:, 0], 0, c[:, 2], color=C_ENR, lw=1.3)
    ax.plot(c[:, 0], c[:, 2], "o", ms=3.0, color=C_ENR, mew=0)
    cl = c[(c[:, 0] > 0.12) & (c[:, 0] < 0.20)]
    cl = cl[np.argsort(cl[:, 0])]
    kmin = int(np.argmin(np.diff(cl[:, 0])))      # the ACTUAL closest pair
    p0, p1 = cl[kmin], cl[kmin + 1]
    dsep = float(p1[0] - p0[0])
    ymax = c[:, 2].max() * 1.45
    yb = ymax * 0.74
    xb = p0[0] - 0.5 * (dfbin - dsep)             # bin centred on the pair
    ax.hlines(yb, xb, xb + dfbin, color=C_FFT, lw=1.0)
    ax.vlines([xb, xb + dfbin], yb - ymax * 0.025, yb + ymax * 0.025,
              color=C_FFT, lw=1.0)
    print(f"  panel f: DFT grid step 1/L = {dfbin:.4f}, closest pair "
          f"{p0[0]:.4f} / {p1[0]:.4f}, separation {dsep:.4f}")
    for pk in (p0, p1):
        ax.annotate("", xy=(pk[0], pk[2] + ymax * 0.03),
                    xytext=(pk[0], yb - ymax * 0.04),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=C_FFT, ls=":"))
    ax.set_xlim(0.118, 0.238); ax.set_ylim(0, ymax)
    ax.set_xlabel("recovered frequency  (cycles/time)")
    ax.set_ylabel("amplitude  $|W_2^{\\,j}|$")
    panel_letter(ax, "f", dx=-0.28)
    print(f"  islet cluster f: {np.round(cl[:, 0], 4)}"
          f" | pairwise gaps {np.round(np.diff(cl[:, 0]), 5)}"
          f" | CLOSEST PAIR ({p0[0]:.4f}, {p1[0]:.4f}) gap {dsep:.5f}"
          f" | DFT bin {dfbin:.5f}", flush=True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(here, f"fig_missing.{ext}"), dpi=300,
                    bbox_inches="tight")
    print("wrote fig_missing.pdf / .png")


if __name__ == "__main__":
    main()
