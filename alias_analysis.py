"""
Aliasing and identifiability of the SinENR parameterization.

THEORY.  On a uniform grid x_n = x_0 + n*Delta, a sine atom is invariant under a
discrete group acting on (omega, phi, a).  With t_n = n*Delta,

    sin(omega t_n + phi)  where omega = 2*pi*f

    (i)  f -> f + k/Delta            leaves every sample unchanged:
         sin((omega + 2*pi*k/Delta) n Delta + phi) = sin(omega t_n + phi + 2*pi*k*n)
    (ii) f -> k/Delta - f            leaves every sample unchanged up to
         (a, phi) -> (-a, -phi):
         sin((2*pi*k/Delta - omega) n Delta + phi) = -sin(omega t_n - phi)

So the orbit of f under the alias group is  { +/- f + k*f_s },  f_s = 1/Delta.
Members of one orbit are INDISTINGUISHABLE IN THE TRAINING LOSS -- not merely
close, but equal to machine precision.  The ENR readout (W1, b1, W2) is
therefore identifiable only modulo this group.  Two things break the tie:

    * restricting the frequency prior / band penalty to omega_max < pi/Delta
      (the Nyquist limit) -- one orbit member per band; or
    * non-uniform sampling, under which (i) and (ii) no longer hold exactly.

Multi-start training does NOT break the tie: restarts are selected on validation
loss, which is in-sample and hence alias-blind.

This script measures (E1) the degeneracy, (E2) which basin initialization lands
in as a function of the prior bound, (E3) the effect of non-uniform sampling.

Usage:  python alias_analysis.py   -> results_alias.csv, alias_landscape.npz
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.append(os.getcwd())
from sinenr import (SinENR, fit_sinenr, predict, spectral_readout, rmse,
                    two_tone, _normalize)

F_TRUE = (0.7, 1.5)
N, X_MAX = 100, 3.0
TOL = 0.05                 # cycles: tolerance for assigning a component to a branch
N_SEEDS = 20
EPOCHS, LR, M, LAM = 3000, 1e-2, 100, 5e-3
DENSE = 4000


def grid_facts():
    x, y = two_tone(N=N, x_max=X_MAX, f=F_TRUE)
    dx = float(np.mean(np.diff(x)))
    return x, y, dx, 1.0 / dx, 0.5 / dx          # x, y, Delta, f_s, f_Nyquist


def alias_orbit(f, f_s, f_hi):
    """All members of {+/- f + k f_s} lying in [0, f_hi]."""
    out = []
    k = 0
    while k * f_s - f <= f_hi + 1e-9:
        for cand in (k * f_s + f, k * f_s - f):
            if -1e-9 <= cand <= f_hi + 1e-9:
                out.append(round(float(cand), 9)) if cand not in out else None
        k += 1
    return sorted(set(out))


# ---------------- E1: is the degeneracy exact? ---------------------------
def e1_degeneracy():
    """Evaluate the training MSE at the true solution and at its aliases."""
    x, y, dx, f_s, f_nyq = grid_facts()
    xn, x_min, x_scale = _normalize(x)
    L = float(np.ravel(x_scale)[0]); x_min = float(np.ravel(x_min)[0])
    X = torch.tensor(xn, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.float64)

    rows = []
    for k in range(0, 4):
        for sign in (+1, -1):
            # build the M=2 atom pair at the aliased frequencies
            f_al = [sign * f + k * f_s for f in F_TRUE]
            if min(f_al) < 0:
                continue
            m = SinENR(2, omega_max=1e9, generator=torch.Generator().manual_seed(0)).double()
            with torch.no_grad():
                m.W1[:, 0] = torch.tensor([2 * np.pi * f * L for f in f_al])
                # alias (ii) flips amplitude and phase
                m.b1[:] = torch.tensor([0.0, 0.0]) if sign > 0 else torch.tensor([0.0, 0.0])
                m.W2[:] = torch.tensor([1.0, 1.0]) if sign > 0 else torch.tensor([-1.0, -1.0])
                m.b2[:] = 0.0
                mse = float(torch.mean((m(X) - Y) ** 2))
                # out-of-sample behaviour on a dense grid
                xd = np.linspace(0, X_MAX, DENSE)
                yd = sum(np.sin(2 * np.pi * f * xd) for f in F_TRUE)
                xdn, _, _ = _normalize(xd, np.array([x_min]), np.array([L]))
                pred = m(torch.tensor(xdn, dtype=torch.float64)).numpy()
            rows.append(dict(k=k, sign=sign, f_alias=str([round(f, 4) for f in f_al]),
                             in_sample_mse=mse, dense_rmse=rmse(pred, yd),
                             above_nyquist=bool(max(f_al) > f_nyq)))
    return pd.DataFrame(rows)


# ---------------- E1b: loss profile along W1 -----------------------------
def e1_landscape(jitter=0.0, seed=0):
    """MSE vs the frequency of one atom, other atom pinned at the second tone."""
    x, y, dx, f_s, f_nyq = grid_facts()
    if jitter > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-jitter * dx, jitter * dx, len(x))
        x[0], x[-1] = 0.0, X_MAX
        x = np.sort(x)
        y = sum(np.sin(2 * np.pi * f * x) for f in F_TRUE)
    xn, x_min, x_scale = _normalize(x)
    L = float(np.ravel(x_scale)[0]); x_min = float(np.ravel(x_min)[0])
    X = torch.tensor(xn, dtype=torch.float64)
    Y = torch.tensor(y, dtype=torch.float64)
    f_scan = np.linspace(0.0, 2.2 * f_s, 9000)
    m = SinENR(2, omega_max=1e9, generator=torch.Generator().manual_seed(0)).double()
    with torch.no_grad():
        m.b1[:] = 0.0; m.W2[:] = 1.0; m.b2[:] = 0.0
        m.W1[1, 0] = 2 * np.pi * F_TRUE[1] * L       # second atom pinned at f = 1.5
        mse = []
        for f in f_scan:
            m.W1[0, 0] = 2 * np.pi * f * L
            mse.append(float(torch.mean((m(X) - Y) ** 2)))
    return f_scan, np.array(mse), f_s, f_nyq


# ---------------- E2 / E3: which basin does initialization find? ---------
def classify(f_hat, f_s, f_hi):
    """'true' | 'alias' | 'other' for one recovered frequency."""
    for f0 in F_TRUE:
        if abs(f_hat - f0) <= TOL:
            return "true"
    for f0 in F_TRUE:
        for fa in alias_orbit(f0, f_s, f_hi):
            if min(abs(f_hat - f0) for f0 in F_TRUE) > TOL and abs(f_hat - fa) <= TOL:
                return "alias"
    return "other"


def e2_basins(f_max_list=(2.5, None, 33.0, 66.0), jitter=0.0, tag="uniform"):
    x0, y0, dx, f_s, f_nyq = grid_facts()
    xd = np.linspace(0, X_MAX, DENSE)
    yd = sum(np.sin(2 * np.pi * f * xd) for f in F_TRUE)
    rows = []
    for f_max in f_max_list:
        fm = f_nyq if f_max is None else f_max      # None -> exactly Nyquist
        n_branch = sum(len(alias_orbit(f0, f_s, fm)) for f0 in F_TRUE) / len(F_TRUE)
        for seed in range(1, N_SEEDS + 1):
            x, y = x0.copy(), y0.copy()
            if jitter > 0:
                rng = np.random.default_rng(50_000 + seed)
                x = x + rng.uniform(-jitter * dx, jitter * dx, len(x))
                x[0], x[-1] = 0.0, X_MAX
                x = np.sort(x)
                y = sum(np.sin(2 * np.pi * f * x) for f in F_TRUE)
            L = float(np.ptp(x))
            f_ = fit_sinenr(x, y, n_units=M, omega_max=2 * np.pi * fm * L, lam_l1=LAM,
                            epochs=EPOCHS, lr=LR, n_restarts=3, seed=seed)
            c = spectral_readout(f_)
            kinds = [classify(ci, f_s, fm) for ci in c[:, 0]]
            rows.append(dict(sampling=tag, f_max=fm, f_max_over_nyq=fm / f_nyq,
                             branches_per_tone=n_branch, seed=seed, n_comp=len(c),
                             n_true=kinds.count("true"), n_alias=kinds.count("alias"),
                             n_other=kinds.count("other"),
                             in_sample_mse=f_.val_loss,
                             dense_rmse=rmse(predict(f_, xd), yd),
                             f_recovered=";".join(f"{v:.4f}" for v in c[:, 0])))
        print(f"  [{tag}] f_max = {fm:.3g} done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    x, y, dx, f_s, f_nyq = grid_facts()
    print(f"grid: Delta = {dx:.6f}, f_s = {f_s:.4f}, f_Nyquist = {f_nyq:.4f}", flush=True)
    for f0 in F_TRUE:
        print(f"  alias orbit of f = {f0} up to 2 f_s: {alias_orbit(f0, f_s, 2*f_s)}", flush=True)

    d1 = e1_degeneracy(); d1.to_csv("results_alias_degeneracy.csv", index=False)
    print(d1.to_string(index=False), flush=True)

    fs_u, mse_u, _, _ = e1_landscape(jitter=0.0)
    fs_j, mse_j, _, _ = e1_landscape(jitter=0.4, seed=7)
    np.savez("alias_landscape.npz", f_scan=fs_u, mse_uniform=mse_u, mse_jitter=mse_j,
             f_s=f_s, f_nyq=f_nyq)
    print(f"[{time.time()-t0:5.0f}s] landscapes done", flush=True)

    r = pd.concat([e2_basins(tag="uniform"),
                   e2_basins(f_max_list=(66.0,), jitter=0.4, tag="jittered 40%")],
                  ignore_index=True)
    r.to_csv("results_alias.csv", index=False)
    print(f"[{time.time()-t0:5.0f}s] basins done", flush=True)
    print(r.groupby(["sampling", "f_max"])[["n_true", "n_alias", "n_other", "n_comp",
                                            "in_sample_mse", "dense_rmse"]]
           .mean().to_string(float_format=lambda v: f"{v:.4g}"), flush=True)
