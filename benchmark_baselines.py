"""
Parametric line-spectral estimators on the two manuscript benchmarks.

Added in revision (Referee 1, point 1).  The estimators of
`spectral_baselines.py` are given exactly the data, the model order and the
admissible band that SinENR is given in `benchmark.py`, and are scored with
the same metrics, so the rows can be placed directly in Tables II and IV.

  task A  clean two-tone,  f = {0.7, 1.5},  N = 100, x in [0, 3],  K = 2
          deterministic: one run per estimator.
  task C  four-tone,  f = {5.0, 13.3, 21.7, 30.0},  sigma = 1,
          N = 512, x in [0, 1],  K = 4; 10 noise seeds, identical to
          benchmark.py (rng = default_rng(20000 + seed)).

Usage:  python benchmark_baselines.py
        -> results/results_baselines_raw.csv, results/table_baselines.csv
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinenr import circ_err, match_to_truth, multi_tone, rmse, two_tone
from spectral_baselines import ESTIMATORS, reconstruct

# ---- benchmark constants, copied verbatim from benchmark.py --------------
N_SEEDS = 10
F_A, X_MAX_A, N_A = (0.7, 1.5), 3.0, 100
F_C, X_MAX_C, N_C, SIGMA_C = (5.0, 13.3, 21.7, 30.0), 1.0, 512, 1.0
PHI_C = np.random.default_rng(0).uniform(-np.pi, np.pi, len(F_C))
DENSE = 2000
FMAX_A, FMAX_C = 2.5, 40.0          # the admissible bands SinENR was given

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _score(comps, f_true, phi_true, amp_true):
    m = match_to_truth(comps, f_true)
    return dict(
        f_err=float(np.nanmean(np.abs(m[:, 0] - np.asarray(f_true)))),
        phase_err=float(np.nanmean(circ_err(m[:, 1], phi_true))),
        amp_err=float(np.nanmean(np.abs(m[:, 2] - amp_true))),
    )


def task_A(rows):
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    xd = np.linspace(0, X_MAX_A, DENSE)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_A)
    phi_true, amp_true = np.zeros(len(F_A)), np.ones(len(F_A))
    for name, fn in ESTIMATORS.items():
        t0 = time.time()
        c = fn(x, y, K=len(F_A), f_max=FMAX_A)
        rows.append(dict(task="A", method=name, seed=0, n_comp=len(c),
                         recon_rmse=rmse(reconstruct(c, xd), yd),
                         runtime_s=time.time() - t0,
                         **_score(c, F_A, phi_true, amp_true)))


def task_B(rows, frac_missing=0.30):
    """Missing-data task.  Only NLS is applicable: MUSIC, ESPRIT and the
    matrix pencil are built on shift invariance of a uniform grid and have no
    definition on a gapped record, whereas NLS is ordinary regression on
    whatever coordinates are observed."""
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    n_mask = int(round(frac_missing * len(x)))
    # mask streams replicated exactly from benchmark.py (isolated) and
    # benchmark_extra.py task_B2 (contiguous), so every method sees the same
    # withheld indices seed by seed.
    geoms = [("30x1 isolated", None), ("3x10 blocks", 3), ("1x30 block", 1)]
    for gname, nb in geoms:
        for seed in range(1, N_SEEDS + 1):
            if nb is None:
                rng = np.random.default_rng(10_000 + seed)
                mask = np.sort(rng.choice(np.arange(1, len(x) - 1), n_mask,
                                          replace=False))
            else:
                blk = n_mask // nb
                rng = np.random.default_rng(30_000 + 100 * nb + seed)
                starts, tries = [], 0
                while len(starts) < nb and tries < 1000:
                    s = int(rng.integers(1, len(x) - blk - 1))
                    cand = set(range(s, s + blk))
                    if all(not (cand & set(range(m, m + blk))) for m in starts):
                        starts.append(s)
                    tries += 1
                mask = np.sort(np.concatenate([np.arange(m, m + blk) for m in starts]))
            obs = np.setdiff1d(np.arange(len(x)), mask)
            t0 = time.time()
            c = ESTIMATORS["NLS (known K)"](x[obs], y[obs], K=len(F_A), f_max=FMAX_A)
            rows.append(dict(task="B", geometry=gname, method="NLS (known K)",
                             seed=seed, n_comp=len(c),
                             gap_rmse=rmse(reconstruct(c, x[mask]), y[mask]),
                             runtime_s=time.time() - t0,
                             **_score(c, F_A, np.zeros(len(F_A)), np.ones(len(F_A)))))


def task_C(rows):
    x, clean = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
    amp_true = np.ones(len(F_C))
    for seed in range(1, N_SEEDS + 1):
        rng = np.random.default_rng(20_000 + seed)
        noisy = clean + rng.normal(0, SIGMA_C, len(x))
        for name, fn in ESTIMATORS.items():
            t0 = time.time()
            c = fn(x, noisy, K=len(F_C), f_max=FMAX_C)
            rows.append(dict(task="C", method=name, seed=seed, n_comp=len(c),
                             denoise_rmse=rmse(reconstruct(c, x), clean),
                             runtime_s=time.time() - t0,
                             **_score(c, F_C, PHI_C, amp_true)))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    t0 = time.time()
    task_A(rows)
    print(f"[{time.time()-t0:6.1f}s] task A done", flush=True)
    task_B(rows)
    print(f"[{time.time()-t0:6.1f}s] task B done", flush=True)
    task_C(rows)
    print(f"[{time.time()-t0:6.1f}s] task C done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results_baselines_raw.csv"), index=False)

    metrics = [c for c in df.columns if c not in ("task", "geometry", "method", "seed")]
    df["geometry"] = df.get("geometry", pd.Series(index=df.index, dtype=object)).fillna("-")
    summ = (df.groupby(["task", "geometry", "method"])[metrics]
              .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(os.path.join(OUT, "table_baselines.csv"), index=False)
    print(summ.to_string(), flush=True)


if __name__ == "__main__":
    main()
