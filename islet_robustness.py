"""
Is the close frequency pair in the islet readout a pair of physical modes?

Added in revision (Referee 1, point 4).  Figure 4(f) of the submitted version
reported two retained components at f = 0.180 and 0.191 cycles per unit time,
separated by less than the DFT grid spacing 1/L of the same record, and read
them as two resolved rhythms.  The islet trace is not a line spectrum: it comes
from a quasi-periodic, amplitude-modulated nonlinear system, so a pair of
sinusoidal atoms may equally well be two atoms approximating one modulated
component.  Five independent checks are run here.

  1. natural frequencies   -- the distribution of the oscillators' own
                              frequencies OM/2*pi actually present in the
                              simulation that produced the trace.
  2. long-trajectory       -- periodogram of a trajectory 8x longer than the
                              analysed record, whose DFT grid spacing is far
                              below the claimed separation.
  3. record length         -- refit on records of growing length; a physical
                              mode keeps its frequency, an approximation atom
                              need not.
  4. regularization        -- refit over lambda_1 and the unit count M.
  5. bootstrap             -- case-resampling refits of the analysed record.

Usage:  python islet_robustness.py
        -> results/islet_*.csv  and a console summary
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import islet_model as IM
from sinenr import fit_sinenr, spectral_readout

OUT = os.path.join(HERE, "results")

# ---- the configuration of Fig. 4(f), copied from make_fig_missing.py ----
STRIDE, FRAC, SEED_MASK, FIT_SEED = 4, 0.30, 778, 1
N_UNITS, LAM1, EPOCHS, LR, N_RESTARTS, TAU_REL = 100, 1e-3, 3000, 1e-2, 3, 0.05
CLUSTER = (0.12, 0.24)          # the dominant-cluster window of Fig. 4(f)
PAIR = (0.180, 0.191)           # the reported pair


def islet_record(itmax=IM.ITMAX, stride=STRIDE, seed=0, burn_in=0.25):
    t, g, _ = IM.simulate(seed=seed, itmax=itmax, burn_in=burn_in)
    return t[::stride], g[::stride]


def contiguous_mask(n, frac=FRAC, n_blocks=1, seed=SEED_MASK):
    blk = int(round(frac * n)) // n_blocks
    rng = np.random.default_rng(seed)
    starts = []
    while len(starts) < n_blocks:
        s = int(rng.integers(1, n - blk - 1))
        if all(not (set(range(s, s + blk)) & set(range(m, m + blk))) for m in starts):
            starts.append(s)
    return np.sort(np.concatenate([np.arange(m, m + blk) for m in starts]))


def fit_and_read(t, g, lam1=LAM1, n_units=N_UNITS, seed=FIT_SEED, f_max=1.0,
                 tau_rel=TAU_REL):
    L = float(np.ptp(t))
    f_ = fit_sinenr(t, g, seed=seed, n_units=n_units,
                    omega_max=2 * np.pi * f_max * L, lam_l1=lam1,
                    epochs=EPOCHS, lr=LR, n_restarts=N_RESTARTS)
    return spectral_readout(f_, tau_rel=tau_rel)


def cluster_stats(comps, L):
    """Summarize the retained components inside the dominant-cluster window."""
    if len(comps) == 0:
        return dict(n_cluster=0, min_gap=np.nan, centroid=np.nan,
                    cluster_amp=np.nan, n_below_bin=0, dft_bin=1.0 / L)
    c = comps[(comps[:, 0] >= CLUSTER[0]) & (comps[:, 0] <= CLUSTER[1])]
    if len(c) == 0:
        return dict(n_cluster=0, min_gap=np.nan, centroid=np.nan,
                    cluster_amp=np.nan, n_below_bin=0, dft_bin=1.0 / L)
    f = np.sort(c[:, 0])
    gaps = np.diff(f) if len(f) > 1 else np.array([np.nan])
    w = c[:, 2]
    return dict(n_cluster=len(c),
                min_gap=float(np.nanmin(gaps)) if len(f) > 1 else np.nan,
                centroid=float(np.average(c[:, 0], weights=w)),
                cluster_amp=float(w.sum()),
                n_below_bin=int(np.sum(gaps < 1.0 / L)) if len(f) > 1 else 0,
                dft_bin=1.0 / L)


# ======================================================================
# 1. natural frequencies actually present in the simulation
# ======================================================================
def check_natural(rows_out):
    rng = np.random.default_rng(0)                 # the stream simulate() uses
    om = 1.0 + 0.1 * rng.standard_normal((3, IM.NMAX))
    f_nat = om / (2.0 * np.pi)
    df = pd.DataFrame(dict(cell=np.repeat(["alpha", "beta", "delta"], IM.NMAX),
                           f_natural=f_nat.ravel()))
    df.to_csv(os.path.join(OUT, "islet_natural_frequencies.csv"), index=False)
    lo, hi = np.quantile(f_nat, [0.025, 0.975])
    rows_out.append(dict(check="natural frequencies", quantity="mean",
                         value=float(f_nat.mean())))
    rows_out.append(dict(check="natural frequencies", quantity="sd",
                         value=float(f_nat.std())))
    rows_out.append(dict(check="natural frequencies", quantity="q2.5",
                         value=float(lo)))
    rows_out.append(dict(check="natural frequencies", quantity="q97.5",
                         value=float(hi)))
    in_band = float(np.mean((f_nat >= CLUSTER[0]) & (f_nat <= CLUSTER[1])))
    rows_out.append(dict(check="natural frequencies",
                         quantity="fraction inside cluster window", value=in_band))
    print(f"[1] natural f = {f_nat.mean():.4f} +- {f_nat.std():.4f} "
          f"cycles/time; 95% range [{lo:.4f}, {hi:.4f}]; "
          f"{100*in_band:.0f}% inside the cluster window {CLUSTER}", flush=True)
    return f_nat


# ======================================================================
# 2. long-trajectory periodogram
# ======================================================================
def check_longrun(rows_out, factor=8):
    t_s, g_s = islet_record()                      # the analysed record
    L_s = float(np.ptp(t_s))
    t_l, g_l = islet_record(itmax=IM.ITMAX * factor)
    L_l = float(np.ptp(t_l))
    dx = float(np.mean(np.diff(t_l)))

    Y = np.abs(np.fft.rfft(g_l - g_l.mean()))
    fg = np.fft.rfftfreq(len(g_l), d=dx)
    sel = (fg >= CLUSTER[0]) & (fg <= CLUSTER[1])
    pd.DataFrame(dict(f=fg[sel], amplitude=2.0 * Y[sel] / len(g_l))).to_csv(
        os.path.join(OUT, "islet_longrun_spectrum.csv"), index=False)

    # how many resolved local maxima does the long record show in the window?
    m = 2.0 * Y[sel] / len(g_l)
    fs = fg[sel]
    peaks = [i for i in range(1, len(m) - 1) if m[i] > m[i - 1] and m[i] > m[i + 1]]
    peaks.sort(key=lambda i: -m[i])
    strong = [i for i in peaks if m[i] >= 0.25 * m[peaks[0]]] if peaks else []
    f_peak = float(fs[peaks[0]]) if peaks else np.nan

    for q, v in [("record length analysed", L_s), ("record length long", L_l),
                 ("dft bin analysed", 1.0 / L_s), ("dft bin long", 1.0 / L_l),
                 ("dominant peak", f_peak),
                 ("peaks >= 25% of max in window", len(strong))]:
        rows_out.append(dict(check="long trajectory", quantity=q, value=float(v)))
    print(f"[2] long run: L = {L_l:.0f} (bin {1/L_l:.5f}, analysed record bin "
          f"{1/L_s:.5f}); dominant peak {f_peak:.4f}; "
          f"{len(strong)} peaks >= 25% of max inside {CLUSTER}", flush=True)
    return fs, m, strong


# ======================================================================
# 3. record length  /  4. regularization and model order
# ======================================================================
def check_fits(rows_out, comp_rows=None):
    fits = []
    t_full, g_full = islet_record(itmax=IM.ITMAX * 4)      # long enough to slice

    # --- 3. growing record length, same sampling, no mask ---------------
    for n in (188, 281, 375, 563, 750, 1125):
        if n > len(t_full):
            continue
        t, g = t_full[:n], g_full[:n]
        L = float(np.ptp(t))
        t0 = time.time()
        c = fit_and_read(t, g)
        st = cluster_stats(c, L)
        fits.append(dict(check="record length", condition=f"N={n}", N=n,
                         L=L, n_retained=len(c), runtime_s=time.time() - t0, **st))
        _log_components(comp_rows, "record length", f"N={n}", c, L)
        print(f"[3] N={n:5d} L={L:6.1f} bin={1/L:.5f} -> {len(c):2d} retained, "
              f"{st['n_cluster']} in cluster, min gap "
              f"{st['min_gap']:.5f}, centroid {st['centroid']:.4f}", flush=True)

    # --- 4. lambda_1 and unit count on the analysed (masked) record -----
    t_, g_ = islet_record()
    mi = contiguous_mask(len(t_))
    obs = np.setdiff1d(np.arange(len(t_)), mi)
    L = float(np.ptp(t_))
    for lam in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
        t0 = time.time()
        c = fit_and_read(t_[obs], g_[obs], lam1=lam)
        st = cluster_stats(c, L)
        fits.append(dict(check="lambda_1", condition=f"lam={lam:g}", lam1=lam,
                         L=L, n_retained=len(c), runtime_s=time.time() - t0, **st))
        _log_components(comp_rows, "lambda_1", f"lam={lam:g}", c, L)
        print(f"[4a] lam={lam:7.1e} -> {len(c):2d} retained, {st['n_cluster']} in "
              f"cluster, min gap {st['min_gap']:.5f}, centroid {st['centroid']:.4f}",
              flush=True)
    for M in (50, 100, 200):
        t0 = time.time()
        c = fit_and_read(t_[obs], g_[obs], n_units=M)
        st = cluster_stats(c, L)
        fits.append(dict(check="model order", condition=f"M={M}", n_units=M,
                         L=L, n_retained=len(c), runtime_s=time.time() - t0, **st))
        _log_components(comp_rows, "model order", f"M={M}", c, L)
        print(f"[4b] M={M:4d} -> {len(c):2d} retained, {st['n_cluster']} in cluster, "
              f"min gap {st['min_gap']:.5f}, centroid {st['centroid']:.4f}", flush=True)

    df = pd.DataFrame(fits)
    df.to_csv(os.path.join(OUT, "islet_robustness_fits.csv"), index=False)
    return df


# ======================================================================
# 5. bootstrap on the analysed record
# ======================================================================
def _log_components(comp_rows, check, condition, comps, L):
    """Record every retained component, flagging those in the cluster window."""
    if comp_rows is None:
        return
    for f, phi, a in np.asarray(comps, dtype=np.float64):
        comp_rows.append(dict(check=check, condition=condition, f=float(f),
                              phase=float(phi), amplitude=float(a), L=L,
                              in_cluster=bool(CLUSTER[0] <= f <= CLUSTER[1])))


def check_bootstrap(rows_out, n_boot=40, comp_rows=None):
    t_, g_ = islet_record()
    mi = contiguous_mask(len(t_))
    obs = np.setdiff1d(np.arange(len(t_)), mi)
    L = float(np.ptp(t_))
    rng = np.random.default_rng(4242)
    rows = []
    for b in range(n_boot):
        idx = np.sort(rng.integers(0, len(obs), len(obs)))
        c = fit_and_read(t_[obs][idx], g_[obs][idx], seed=100 + b)
        st = cluster_stats(c, L)
        # nearest retained component to each member of the reported pair
        near = {}
        for lbl, f0 in zip(("lo", "hi"), PAIR):
            near[f"f_near_{lbl}"] = (float(c[np.argmin(np.abs(c[:, 0] - f0)), 0])
                                     if len(c) else np.nan)
        rows.append(dict(boot=b, n_retained=len(c), **st, **near))
        _log_components(comp_rows, "bootstrap", f"boot={b}", c, L)
        if (b + 1) % 10 == 0:
            print(f"[5] bootstrap {b+1}/{n_boot}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "islet_bootstrap_cluster.csv"), index=False)
    frac_two = float(np.mean(df.n_cluster >= 2))
    print(f"[5] bootstrap: n_cluster = {df.n_cluster.mean():.2f} +- "
          f"{df.n_cluster.std():.2f}; >=2 components in {100*frac_two:.0f}% of "
          f"refits; centroid {df.centroid.mean():.4f} +- {df.centroid.std():.4f}; "
          f"min gap {df.min_gap.mean():.5f} +- {df.min_gap.std():.5f}", flush=True)
    for q, v in [("n_cluster mean", df.n_cluster.mean()),
                 ("n_cluster sd", df.n_cluster.std()),
                 ("fraction with >=2 in cluster", frac_two),
                 ("centroid mean", df.centroid.mean()),
                 ("centroid sd", df.centroid.std()),
                 ("min gap mean", df.min_gap.mean()),
                 ("min gap sd", df.min_gap.std()),
                 ("f near 0.180 sd", df.f_near_lo.std()),
                 ("f near 0.191 sd", df.f_near_hi.std())]:
        rows_out.append(dict(check="bootstrap", quantity=q, value=float(v)))
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    summary, comps = [], []
    t0 = time.time()
    check_natural(summary)
    check_longrun(summary)
    check_fits(summary, comp_rows=comps)
    check_bootstrap(summary, comp_rows=comps)
    pd.DataFrame(summary).to_csv(
        os.path.join(OUT, "islet_robustness_summary.csv"), index=False)
    pd.DataFrame(comps).to_csv(
        os.path.join(OUT, "islet_cluster_components.csv"), index=False)
    print(f"done in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
