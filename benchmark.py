"""
Reproducible benchmark for SinENR (explicitly-parameterized neural
representation with sinusoidal basis), against Fourier and spline baselines.

Tasks (signals are those specified in the ENR manuscript itself):
  A  spectral recovery  -- two-tone, f = {0.7, 1.5}, N = 100, x in [0, 3]   (Sec. IV)
  B  missing data       -- same signal, 30 % of samples masked at random    (Sec. V)
  C  denoising          -- four-tone, f = {5.0, 13.3, 21.7, 30.0}, sigma = 1 (Sec. V)

Every task is run over N_SEEDS seeds.  The seed controls initialization and the
validation split in all tasks, the mask in task B, and the noise draw in task C.
The l1 weight is selected once, on seed 0, by the rule "smallest lambda_1 whose
pruned readout yields at most K components", where K is the known model order --
the same information the peak-picking Fourier baseline is given.

Usage:  python benchmark.py            (writes results_raw.csv, results_summary.csv,
                                        lambda_selection.csv, sparsity_tradeoff.csv)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.append(os.getcwd())
from sinenr import (fit_sinenr, predict, spectral_readout, fft_peaks,
                    fft_hard_threshold, rmse, circ_err, match_to_truth,
                    two_tone, multi_tone)

N_SEEDS = 10
LAM_GRID = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
DENSE = 2000          # points for the dense reconstruction-error grid

# ---- task definitions ----------------------------------------------------
F_A = (0.7, 1.5)
X_MAX_A, N_A = 3.0, 100
F_C = (5.0, 13.3, 21.7, 30.0)
X_MAX_C, N_C, SIGMA_C = 1.0, 512, 1.0
# phases of the four-tone signal: drawn once from a fixed seed, then held fixed
PHI_C = np.random.default_rng(0).uniform(-np.pi, np.pi, len(F_C))

CFG = {
    "A": dict(n_units=100, epochs=3000, lr=1e-2, n_restarts=3, f_max=2.5),
    "B": dict(n_units=100, epochs=3000, lr=1e-2, n_restarts=3, f_max=2.5),
    "C": dict(n_units=100, epochs=6000, lr=1e-2, n_restarts=3, f_max=40.0),
}


def omega_max(task, L):
    return 2 * np.pi * CFG[task]["f_max"] * L


def fit_kw(task, L, lam):
    c = CFG[task]
    return dict(n_units=c["n_units"], omega_max=omega_max(task, L), lam_l1=lam,
                epochs=c["epochs"], lr=c["lr"], n_restarts=c["n_restarts"])


# ---- lambda_1 selection --------------------------------------------------
def select_lambda(task, x, y, K, L, record):
    """Smallest lambda_1 whose pruned readout yields <= K components (seed 0)."""
    chosen = None
    for lam in LAM_GRID:
        f_ = fit_sinenr(x, y, seed=0, **fit_kw(task, L, lam))
        c = spectral_readout(f_)
        record.append(dict(task=task, lam=lam, n_comp=len(c), val_mse=f_.val_loss))
        if chosen is None and len(c) <= K:
            chosen = lam
    return chosen if chosen is not None else LAM_GRID[-1]


# ---- task A: spectral recovery ------------------------------------------
def task_A(lam, rows):
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    xd = np.linspace(0, X_MAX_A, DENSE)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_A)
    phi_true = np.zeros(len(F_A))
    amp_true = np.ones(len(F_A))

    for seed in range(1, N_SEEDS + 1):
        f_ = fit_sinenr(x, y, seed=seed, **fit_kw("A", L, lam))
        c = spectral_readout(f_)
        m = match_to_truth(c, F_A)
        rows.append(dict(task="A", method="SinENR", seed=seed, n_comp=len(c),
                         recon_rmse=rmse(predict(f_, xd), yd),
                         f_err=float(np.nanmean(np.abs(m[:, 0] - np.array(F_A)))),
                         phase_err=float(np.nanmean(circ_err(m[:, 1], phi_true))),
                         amp_err=float(np.nanmean(np.abs(m[:, 2] - amp_true)))))

    # Fourier baselines are deterministic -> one row each, replicated for the table
    for pad, name in [(1, "FFT (peak-pick)"), (8, "FFT (8x zero-pad)")]:
        c = fft_peaks(x, y, top_k=len(F_A), pad=pad)
        c = c[np.argsort(c[:, 0])] if len(c) else c
        m = match_to_truth(c, F_A)
        # reconstruction from the retained peaks
        rec = sum(m[k, 2] * np.sin(2 * np.pi * m[k, 0] * xd + m[k, 1])
                  for k in range(len(F_A)) if np.isfinite(m[k, 0]))
        rows.append(dict(task="A", method=name, seed=0, n_comp=len(c),
                         recon_rmse=rmse(rec, yd),
                         f_err=float(np.nanmean(np.abs(m[:, 0] - np.array(F_A)))),
                         phase_err=float(np.nanmean(circ_err(m[:, 1], phi_true))),
                         amp_err=float(np.nanmean(np.abs(m[:, 2] - amp_true)))))


# ---- task B: missing-data reconstruction --------------------------------
def task_B(lam, rows, frac_missing=0.30):
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(10_000 + seed)
        n_mask = int(round(frac_missing * len(x)))
        # never mask the endpoints: interpolation baselines cannot extrapolate
        mask = np.sort(rng.choice(np.arange(1, len(x) - 1), n_mask, replace=False))
        obs = np.setdiff1d(np.arange(len(x)), mask)

        f_ = fit_sinenr(x[obs], y[obs], seed=seed, **fit_kw("B", L, lam))
        rows.append(dict(task="B", method="SinENR", seed=seed,
                         gap_rmse=rmse(predict(f_, x[mask]), y[mask]),
                         n_comp=len(spectral_readout(f_))))
        cs = CubicSpline(x[obs], y[obs])
        rows.append(dict(task="B", method="Cubic spline", seed=seed,
                         gap_rmse=rmse(cs(x[mask]), y[mask]), n_comp=np.nan))
        rows.append(dict(task="B", method="Linear interp", seed=seed,
                         gap_rmse=rmse(np.interp(x[mask], x[obs], y[obs]), y[mask]),
                         n_comp=np.nan))


# ---- task C: denoising ---------------------------------------------------
def task_C(lam, rows):
    x, clean = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
    L = float(np.ptp(x))
    amp_true = np.ones(len(F_C))
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(20_000 + seed)
        noisy = clean + rng.normal(0, SIGMA_C, len(x))

        f_ = fit_sinenr(x, noisy, seed=seed, **fit_kw("C", L, lam))
        c = spectral_readout(f_)
        m = match_to_truth(c, F_C)
        rows.append(dict(task="C", method="SinENR", seed=seed, n_comp=len(c),
                         denoise_rmse=rmse(predict(f_, x), clean),
                         f_err=float(np.nanmean(np.abs(m[:, 0] - np.array(F_C)))),
                         phase_err=float(np.nanmean(circ_err(m[:, 1], PHI_C))),
                         amp_err=float(np.nanmean(np.abs(m[:, 2] - amp_true)))))

        rec = fft_hard_threshold(noisy, n_keep=len(F_C))
        cf = fft_peaks(x, noisy, top_k=len(F_C), pad=1)
        cf = cf[np.argsort(cf[:, 0])] if len(cf) else cf
        mf = match_to_truth(cf, F_C)
        rows.append(dict(task="C", method="FFT hard-threshold", seed=seed,
                         n_comp=len(F_C), denoise_rmse=rmse(rec, clean),
                         f_err=float(np.nanmean(np.abs(mf[:, 0] - np.array(F_C)))),
                         phase_err=float(np.nanmean(circ_err(mf[:, 1], PHI_C))),
                         amp_err=float(np.nanmean(np.abs(mf[:, 2] - amp_true)))))
        rows.append(dict(task="C", method="No denoising", seed=seed, n_comp=np.nan,
                         denoise_rmse=rmse(noisy, clean)))


# ---- sparsity / fidelity tradeoff (task A, all lambdas, 3 seeds) ---------
def tradeoff(rows):
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    xd = np.linspace(0, X_MAX_A, DENSE)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_A)
    for lam in LAM_GRID:
        for seed in range(1, 4):
            f_ = fit_sinenr(x, y, seed=seed, **fit_kw("A", L, lam))
            c = spectral_readout(f_)
            m = match_to_truth(c, F_A)
            rows.append(dict(lam=lam, seed=seed, n_comp=len(c),
                             recon_rmse=rmse(predict(f_, xd), yd),
                             f_err=float(np.nanmean(np.abs(m[:, 0] - np.array(F_A)))),
                             amp_err=float(np.nanmean(np.abs(m[:, 2] - 1.0)))))


def main():
    t0 = time.time()
    lam_rec, raw, tr = [], [], []

    lam_A = select_lambda("A", *two_tone(N=N_A, x_max=X_MAX_A, f=F_A),
                          K=len(F_A), L=X_MAX_A, record=lam_rec)
    print(f"[{time.time()-t0:6.0f}s] lambda_A = {lam_A}", flush=True)
    xC, cleanC = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
    noisy0 = cleanC + np.random.default_rng(20_001).normal(0, SIGMA_C, N_C)
    lam_C = select_lambda("C", xC, noisy0, K=len(F_C), L=X_MAX_C, record=lam_rec)
    print(f"[{time.time()-t0:6.0f}s] lambda_C = {lam_C}", flush=True)

    task_A(lam_A, raw);  print(f"[{time.time()-t0:6.0f}s] task A done", flush=True)
    task_B(lam_A, raw);  print(f"[{time.time()-t0:6.0f}s] task B done", flush=True)
    task_C(lam_C, raw);  print(f"[{time.time()-t0:6.0f}s] task C done", flush=True)
    tradeoff(tr);        print(f"[{time.time()-t0:6.0f}s] tradeoff done", flush=True)

    df = pd.DataFrame(raw)
    df.to_csv("results_raw.csv", index=False)
    pd.DataFrame(lam_rec).to_csv("lambda_selection.csv", index=False)
    pd.DataFrame(tr).to_csv("sparsity_tradeoff.csv", index=False)

    metrics = [c for c in df.columns if c not in ("task", "method", "seed")]
    summ = (df.groupby(["task", "method"])[metrics]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv("results_summary.csv", index=False)
    with open("lambda_chosen.txt", "w") as f:
        f.write(f"lambda_A={lam_A}\nlambda_C={lam_C}\nphi_C={list(PHI_C)}\n")
    print(f"[{time.time()-t0:6.0f}s] wrote results_raw.csv ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
