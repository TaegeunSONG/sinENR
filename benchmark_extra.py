"""
Two follow-ups to benchmark.py, each triggered by a result in it.

B2  Contiguous-gap reconstruction.  In benchmark.py's task B the 30 % of masked
    points are isolated single samples on a 100-point grid, leaving the 1.5-cycle
    component oversampled ~33x; cubic-spline interpolation is then near-exact and
    the task cannot separate the methods.  B2 masks the same *fraction* as
    contiguous blocks, where a global learned basis should matter.

C2  Extended l1 grid for the noisy four-tone signal.  benchmark.py's selection
    rule ("smallest lambda_1 with at most K components") never reached K = 4
    within its grid, so the sparsity claim was untested at sigma = 1.  C2
    continues the grid upward and records the fidelity cost.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.append(os.getcwd())
from sinenr import (fit_sinenr, predict, spectral_readout, rmse, circ_err,
                    match_to_truth, two_tone, multi_tone)
from benchmark import (F_A, N_A, X_MAX_A, F_C, N_C, X_MAX_C, SIGMA_C, PHI_C,
                       N_SEEDS, fit_kw)


def task_B2(lam=5e-3, frac=0.30, n_blocks=(1, 3)):
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    n_mask = int(round(frac * len(x)))
    rows = []
    for nb in n_blocks:
        blk = n_mask // nb
        for seed in range(1, N_SEEDS + 1):
            rng = np.random.default_rng(30_000 + 100 * nb + seed)
            # non-overlapping contiguous blocks, endpoints always observed
            mask = []
            tries = 0
            while len(mask) < nb and tries < 1000:
                s = int(rng.integers(1, len(x) - blk - 1))
                cand = set(range(s, s + blk))
                if all(not (cand & set(range(m, m + blk))) for m in mask):
                    mask.append(s)
                tries += 1
            midx = np.sort(np.concatenate([np.arange(m, m + blk) for m in mask]))
            obs = np.setdiff1d(np.arange(len(x)), midx)

            f_ = fit_sinenr(x[obs], y[obs], seed=seed, **fit_kw("B", L, lam))
            rows.append(dict(n_blocks=nb, block_len=blk, method="SinENR", seed=seed,
                             gap_rmse=rmse(predict(f_, x[midx]), y[midx]),
                             n_comp=len(spectral_readout(f_))))
            cs = CubicSpline(x[obs], y[obs])
            rows.append(dict(n_blocks=nb, block_len=blk, method="Cubic spline", seed=seed,
                             gap_rmse=rmse(cs(x[midx]), y[midx]), n_comp=np.nan))
            rows.append(dict(n_blocks=nb, block_len=blk, method="Linear interp", seed=seed,
                             gap_rmse=rmse(np.interp(x[midx], x[obs], y[obs]), y[midx]),
                             n_comp=np.nan))
    return pd.DataFrame(rows)


def task_C2(lams=(0.02, 0.05, 0.1, 0.2, 0.4), n_seeds=3):
    x, clean = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
    L = float(np.ptp(x))
    rows = []
    for lam in lams:
        for seed in range(1, n_seeds + 1):
            rng = np.random.default_rng(20_000 + seed)
            noisy = clean + rng.normal(0, SIGMA_C, len(x))
            f_ = fit_sinenr(x, noisy, seed=seed, **fit_kw("C", L, lam))
            c = spectral_readout(f_)
            m = match_to_truth(c, F_C)
            rows.append(dict(lam=lam, seed=seed, n_comp=len(c),
                             denoise_rmse=rmse(predict(f_, x), clean),
                             f_err=float(np.nanmean(np.abs(m[:, 0] - np.array(F_C)))),
                             phase_err=float(np.nanmean(circ_err(m[:, 1], PHI_C))),
                             amp_err=float(np.nanmean(np.abs(m[:, 2] - 1.0)))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    b2 = task_B2(); b2.to_csv("results_B2_contiguous.csv", index=False)
    print(f"[{time.time()-t0:5.0f}s] B2 done", flush=True)
    c2 = task_C2(); c2.to_csv("results_C2_lambda.csv", index=False)
    print(f"[{time.time()-t0:5.0f}s] C2 done", flush=True)
    print(b2.groupby(["n_blocks", "method"])[["gap_rmse"]]
            .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4g}"))
    print(c2.groupby("lam")[["n_comp", "denoise_rmse", "f_err", "phase_err", "amp_err"]]
            .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4g}"))
