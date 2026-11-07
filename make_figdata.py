"""
Build figdata.npz, the per-fit arrays that make_figures.py plots.

The benchmark scripts save per-seed *summary tables*; the figures also need a
few whole trajectories -- one fit's prediction curve, its training history, its
component cloud, a bootstrap ensemble, a gapped-record reconstruction and a
denoising example. Those are regenerated here so that nothing in the figure
pipeline depends on an artifact that is not reproducible from this repository.

Configurations match the benchmark exactly (benchmark.CFG) and the figure
captions: M = 100 units, three restarts, lambda_1 = 5e-3 on the clean two-tone
task and 2e-2 on the denoising task, seed 1 throughout, 60 bootstrap refits.

The loss slice of the landscape panel is the minimal M = 2 model with every
parameter except (W_1^1, W_2^1) pinned at the generating values, swept across
the admissible band [0, omega_max] that the fits are actually given.

Usage:  python make_figdata.py        ->  figdata.npz   (~5 min)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sinenr import (bootstrap_components, canonical_components, fit_sinenr,
                    multi_tone, predict, spectral_readout, two_tone)

# ---- benchmark constants, copied from benchmark.py ----------------------
F_A, X_MAX_A, N_A = (0.7, 1.5), 3.0, 100
F_C, X_MAX_C, N_C, SIGMA_C = (5.0, 13.3, 21.7, 30.0), 1.0, 512, 1.0
PHI_C = np.random.default_rng(0).uniform(-np.pi, np.pi, len(F_C))
DENSE = 2000
SEED = 1
LAM_A, LAM_C = 5e-3, 2e-2
FMAX_A, FMAX_C = 2.5, 40.0
N_BOOT = 60


def kw_A(L, lam=LAM_A):
    return dict(n_units=100, omega_max=2 * np.pi * FMAX_A * L, lam_l1=lam,
                epochs=3000, lr=1e-2, n_restarts=3)


def kw_C(L, lam=LAM_C):
    return dict(n_units=100, omega_max=2 * np.pi * FMAX_C * L, lam_l1=lam,
                epochs=6000, lr=1e-2, n_restarts=3)


def block_mask(n, n_blocks=1, frac=0.30, seed=SEED):
    """The contiguous-block masks of benchmark_extra.task_B2."""
    blk = int(round(frac * n)) // n_blocks
    rng = np.random.default_rng(30_000 + 100 * n_blocks + seed)
    starts, tries = [], 0
    while len(starts) < n_blocks and tries < 1000:
        s = int(rng.integers(1, n - blk - 1))
        if all(not (set(range(s, s + blk)) & set(range(m, m + blk))) for m in starts):
            starts.append(s)
        tries += 1
    return np.sort(np.concatenate([np.arange(m, m + blk) for m in starts]))


def loss_slice(n_w1=420, n_w2=320):
    """MSE over (W_1^1, W_2^1) for the minimal M = 2 model.

    Everything else is pinned at the generating values, so the surface is the
    one the optimizer would see if it had only this unit left to place. The
    sweep covers exactly the admissible band the fits are given,
    omega_max = 2*pi*f_max*L, which is where the alias comb is *not*: on this
    grid consecutive aliases are 2*pi*(N-1) apart, far outside the band.
    """
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    xn = (x - x.min()) / L
    w_true = [2 * np.pi * f * L for f in F_A]          # normalized angular freqs
    fixed = np.sin(w_true[1] * xn)                     # second atom at truth
    omega_max = 2 * np.pi * FMAX_A * L

    w1 = np.linspace(0.0, omega_max, n_w1)
    w2 = np.linspace(-0.2, 2.2, n_w2)
    W1g, W2g = np.meshgrid(w1, w2)
    S = np.sin(w1[None, :] * xn[:, None])              # (N, n_w1)
    R = y[:, None] - fixed[:, None]                    # residual to be matched
    # Z[i, j] = mean( (R - w2_i * S_j)^2 )
    Z = np.empty((n_w2, n_w1))
    for i, a in enumerate(w2):
        Z[i] = np.mean((R - a * S) ** 2, axis=0)
    return W1g, W2g, Z, w_true, omega_max


def main():
    t0 = time.time()
    out = {}

    # ---- clean two-tone: fit, history, components ----------------------
    x, y = two_tone(N=N_A, x_max=X_MAX_A, f=F_A)
    L = float(np.ptp(x))
    xd = np.linspace(0.0, X_MAX_A, DENSE)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_A)
    fitA = fit_sinenr(x, y, seed=SEED, **kw_A(L))
    out.update(xA=x, yA=y, xd=xd, yd=yd,
               pred_A=predict(fitA, xd), hist_A=np.asarray(fitA.history),
               comp_all=canonical_components(fitA),
               comp_kept=spectral_readout(fitA))
    print(f"[{time.time()-t0:5.0f}s] task A fit: {len(out['comp_kept'])} components "
          f"retained of 100", flush=True)

    # ---- loss landscape slice ------------------------------------------
    W1g, W2g, Z, w_true, omega_max = loss_slice()
    out.update(W1g=W1g, W2g=W2g, Z=Z)
    print(f"[{time.time()-t0:5.0f}s] loss slice: W1 in [0, {omega_max:.1f}] "
          f"(true atoms at {w_true[0]:.2f}, {w_true[1]:.2f}); "
          f"MSE spans {np.log10(Z.max()/Z.min()):.1f} decades", flush=True)

    # ---- bootstrap ------------------------------------------------------
    bs = bootstrap_components(x, y, n_boot=N_BOOT, top_k=2, seed=SEED, **kw_A(L))
    out.update(draws=bs["draws"], ref_comp=bs["reference"])
    print(f"[{time.time()-t0:5.0f}s] bootstrap: {N_BOOT} refits", flush=True)

    # ---- contiguous-gap reconstruction ----------------------------------
    midx = block_mask(len(x))
    obs = np.setdiff1d(np.arange(len(x)), midx)
    fitB = fit_sinenr(x[obs], y[obs], seed=SEED, **kw_A(L))
    cs = CubicSpline(x[obs], y[obs])
    out.update(midx=midx, obs=obs, predB=predict(fitB, xd), splineB=cs(xd))
    print(f"[{time.time()-t0:5.0f}s] gapped fit: block of {len(midx)} withheld",
          flush=True)

    # ---- denoising example ----------------------------------------------
    xC, cleanC = multi_tone(N=N_C, x_max=X_MAX_C, f=F_C, phases=PHI_C)
    LC = float(np.ptp(xC))
    noisyC = cleanC + np.random.default_rng(20_000 + SEED).normal(0, SIGMA_C, len(xC))
    fitC = fit_sinenr(xC, noisyC, seed=SEED, **kw_C(LC))
    out.update(xC=xC, cleanC=cleanC, noisyC=noisyC, predC=predict(fitC, xC))
    print(f"[{time.time()-t0:5.0f}s] denoising fit done", flush=True)

    np.savez(os.path.join(HERE, "figdata.npz"), **out)
    print(f"[{time.time()-t0:5.0f}s] wrote figdata.npz ({len(out)} arrays)")


if __name__ == "__main__":
    main()
