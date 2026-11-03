"""
SinENR -- explicitly-parameterized neural representation with sinusoidal basis.

Reference implementation of Eq. (1)-(2) of the ENR manuscript
("Explicitly-Parameterized Neural Representation: An Interpretable Model for
Adaptive Frequency-Phase Signal Decomposition", Kim, Lee, Myoung & Song).

    y_hat_i = b2 + sum_j W2_j * sin( <W1_j, x_i> + b1_j )

Trainable: W1 (frequency-like), b1 (phase), W2 (amplitude), b2 (global bias).

Training follows Sec. III B of the manuscript:
  * inputs normalized to [0, 1]
  * b1 ~ Unif(-pi, pi); W2 small zero-mean; W1 ~ Unif[0, omega_max]
  * Adam + cosine-annealed learning rate
  * l1 penalty on W2 (spectral sparsity)
  * soft frequency bounding (physics-informed constraint: the admissible band)
  * multiple random restarts, best validation loss retained

All randomness is routed through an explicit seed so every number in the
benchmark is reproducible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SinENR(nn.Module):
    """Single-hidden-layer perceptron with sine activation, Eq. (1)."""

    def __init__(self, n_units: int, in_dim: int = 1, omega_max: float = 40.0,
                 amp_scale: float = 1e-2, generator: torch.Generator | None = None):
        super().__init__()
        self.n_units = n_units
        self.in_dim = in_dim
        self.omega_max = float(omega_max)
        g = generator
        # W1 ~ Unif[0, omega_max] (broad prior over the frequency-like weights)
        W1 = torch.rand(n_units, in_dim, generator=g) * omega_max
        # b1 ~ Unif(-pi, pi)
        b1 = (torch.rand(n_units, generator=g) * 2 - 1) * math.pi
        # W2: small zero-mean
        W2 = (torch.rand(n_units, generator=g) * 2 - 1) * amp_scale
        self.W1 = nn.Parameter(W1)
        self.b1 = nn.Parameter(b1)
        self.W2 = nn.Parameter(W2)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        pre = x @ self.W1.T + self.b1          # (N, M)
        return torch.sin(pre) @ self.W2 + self.b2

    # physics-informed term: penalize frequency content outside the admissible
    # band [0, omega_max].  A hard clamp would kill the gradient; this is the
    # "soft frequency bounding" of Sec. III B.
    def band_penalty(self) -> torch.Tensor:
        excess = torch.relu(self.W1.abs() - self.omega_max)
        return (excess ** 2).sum()

    def l1_amplitude(self) -> torch.Tensor:
        return self.W2.abs().sum()


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    model: SinENR
    history: list = field(default_factory=list)   # training loss per epoch (best restart)
    val_loss: float = float("nan")
    restart_val: list = field(default_factory=list)
    x_min: float = 0.0
    x_scale: float = 1.0


def _normalize(x: np.ndarray, x_min=None, x_scale=None):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if x.shape[0] == 1 and x.shape[1] != 1:
        x = x.T
    if x_min is None:
        x_min = x.min(axis=0)
        x_scale = np.ptp(x, axis=0)
        x_scale = np.where(x_scale == 0, 1.0, x_scale)
    return (x - x_min) / x_scale, x_min, x_scale


def fit_sinenr(x, y, n_units=100, omega_max=40.0, lam_l1=0.0, lam_band=1.0,
               epochs=4000, lr=1e-2, n_restarts=5, val_frac=0.15, seed=0,
               amp_scale=1e-2, verbose=False) -> FitResult:
    """Fit Eq. (1) by minimizing the regularized MSE of Eq. (2).

    omega_max is expressed in the *normalized* coordinate (x -> [0,1]):
    omega_max = 2*pi*f_max*(x_range).
    """
    xn, x_min, x_scale = _normalize(x)
    yv = np.asarray(y, dtype=np.float64).ravel()
    N = xn.shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = max(1, int(round(val_frac * N))) if val_frac > 0 else 0
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])

    X = torch.tensor(xn, dtype=torch.float64)
    Y = torch.tensor(yv, dtype=torch.float64)
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = (X[val_idx], Y[val_idx]) if n_val else (X, Y)

    best = None
    restart_val = []
    for r in range(n_restarts):
        g = torch.Generator().manual_seed(int(seed) * 1000 + r)
        model = SinENR(n_units, in_dim=X.shape[1], omega_max=omega_max,
                       amp_scale=amp_scale, generator=g).double()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        hist = []
        for ep in range(epochs):
            opt.zero_grad()
            pred = model(Xtr)
            mse = torch.mean((pred - Ytr) ** 2)
            loss = mse + lam_l1 * model.l1_amplitude() + lam_band * model.band_penalty()
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(mse.detach()))
        with torch.no_grad():
            vl = float(torch.mean((model(Xva) - Yva) ** 2))
        restart_val.append(vl)
        if best is None or vl < best[0]:
            best = (vl, model, hist)
        if verbose:
            print(f"  restart {r}: val MSE = {vl:.3e}")

    vl, model, hist = best
    return FitResult(model=model, history=hist, val_loss=vl,
                     restart_val=restart_val, x_min=float(np.ravel(x_min)[0]),
                     x_scale=float(np.ravel(x_scale)[0]))


def predict(fit: FitResult, x) -> np.ndarray:
    xn, _, _ = _normalize(x, np.array([fit.x_min]), np.array([fit.x_scale]))
    with torch.no_grad():
        return fit.model(torch.tensor(xn, dtype=torch.float64)).numpy()


# --------------------------------------------------------------------------
# interpretable spectral readout (Sec. III A)
# --------------------------------------------------------------------------
def canonical_components(fit: FitResult) -> np.ndarray:
    """Return array of (f, phase, amplitude) in ORIGINAL x units.

    Canonicalization uses sin(-t) = -sin(t) to make omega >= 0, then flips the
    sign of negative amplitudes by adding pi to the phase, so that every
    component is a*sin(2*pi*f*x + phi) with a >= 0 and f >= 0.
    """
    m = fit.model
    W1 = m.W1.detach().numpy().ravel()
    b1 = m.b1.detach().numpy().ravel()
    W2 = m.W2.detach().numpy().ravel()
    L = fit.x_scale
    sgn = np.where(W1 >= 0, 1.0, -1.0)
    omega = np.abs(W1)
    phi = np.where(sgn > 0, b1, -b1)
    amp = W2 * sgn
    # absorb negative amplitude into the phase
    phi = np.where(amp < 0, phi + np.pi, phi)
    amp = np.abs(amp)
    # phase measured at x = x_min; convert normalized frequency to cycles per
    # original x unit.  sin(omega*(x-x_min)/L + phi) = sin(2*pi*f*x + phi')
    f = omega / (2 * np.pi * L)
    phi_abs = phi - omega * fit.x_min / L
    phi_abs = np.mod(phi_abs + np.pi, 2 * np.pi) - np.pi
    return np.stack([f, phi_abs, amp], axis=1)


def _merge(comps: np.ndarray, f_tol: float) -> np.ndarray:
    """Merge near-duplicate components by phasor addition at equal frequency."""
    if len(comps) == 0:
        return comps
    order = np.argsort(comps[:, 0])
    comps = comps[order]
    groups, cur = [], [comps[0]]
    for c in comps[1:]:
        if abs(c[0] - cur[-1][0]) <= f_tol:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for grp in groups:
        g = np.array(grp)
        # a*sin(th + phi) -> Im{ a e^{i phi} e^{i th} }; sum phasors
        z = np.sum(g[:, 2] * np.exp(1j * g[:, 1]))
        f_eff = np.average(g[:, 0], weights=np.maximum(g[:, 2], 1e-300))
        out.append([f_eff, np.angle(z), np.abs(z)])
    return np.array(out)


def spectral_readout(fit: FitResult, tau_rel: float = 0.02, f_tol_rel: float = 0.02,
                     top_k: int | None = None) -> np.ndarray:
    """Pruned + merged spectral summary, sorted by descending amplitude.

    tau_rel   : prune |W2| < tau_rel * max|W2|   (the tau of Sec. III A)
    f_tol_rel : merge tolerance as a fraction of the max retained frequency
    """
    comps = canonical_components(fit)
    if len(comps) == 0:
        return comps
    tau = tau_rel * comps[:, 2].max()
    comps = comps[comps[:, 2] >= tau]
    if len(comps) == 0:
        return comps
    f_tol = f_tol_rel * max(comps[:, 0].max(), 1e-12)
    comps = _merge(comps, f_tol)
    comps = comps[np.argsort(-comps[:, 2])]
    return comps[:top_k] if top_k else comps


# --------------------------------------------------------------------------
# bootstrap uncertainty on the recovered triplets
# --------------------------------------------------------------------------
def bootstrap_components(x, y, n_boot=50, top_k=2, seed=0, warm_start=None,
                         **fit_kw) -> dict:
    """Case-resampling bootstrap; refits and tracks the top-k components.

    Components are matched to the reference fit by nearest frequency.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ref = warm_start if warm_start is not None else fit_sinenr(x, y, seed=seed, **fit_kw)
    ref_comps = spectral_readout(ref, top_k=top_k)
    rng = np.random.default_rng(seed + 991)
    draws = np.full((n_boot, top_k, 3), np.nan)
    for b in range(n_boot):
        idx = np.sort(rng.integers(0, len(x), len(x)))
        fb = fit_sinenr(x[idx], y[idx], seed=seed * 100 + b + 1, **fit_kw)
        cb = spectral_readout(fb)
        if len(cb) == 0:
            continue
        for k in range(min(top_k, len(ref_comps))):
            j = int(np.argmin(np.abs(cb[:, 0] - ref_comps[k, 0])))
            draws[b, k] = cb[j]
    return {"reference": ref_comps, "draws": draws, "fit": ref}


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def fft_peaks(x, y, top_k=2, pad=1):
    """Peak-picked FFT spectrum with parabolic (log-magnitude) interpolation.

    Returns (f, phase, amplitude) for the top_k peaks, matching the
    a*sin(2*pi*f*x + phi) convention used by `spectral_readout`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N = len(y)
    dx = float(np.mean(np.diff(x)))
    nfft = int(N * pad)
    Yf = np.fft.rfft(y - y.mean(), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dx)
    mag = np.abs(Yf)
    # local maxima
    cand = [i for i in range(1, len(mag) - 1) if mag[i] >= mag[i - 1] and mag[i] >= mag[i + 1]]
    cand.sort(key=lambda i: -mag[i])
    out = []
    df = freqs[1] - freqs[0]
    for i in cand[:top_k]:
        a0, a1, a2 = np.log(mag[i - 1] + 1e-300), np.log(mag[i] + 1e-300), np.log(mag[i + 1] + 1e-300)
        denom = (a0 - 2 * a1 + a2)
        delta = 0.5 * (a0 - a2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        f_hat = freqs[i] + delta * df
        amp = 2.0 * mag[i] / N
        phi = np.angle(Yf[i]) + np.pi / 2          # cos-reference -> sin-reference
        phi = float(np.mod(phi + np.pi, 2 * np.pi) - np.pi)
        out.append([f_hat, phi, amp])
    return np.array(out) if out else np.empty((0, 3))


def fft_hard_threshold(y, n_keep):
    """Keep the n_keep largest rFFT coefficients (plus DC), zero the rest."""
    Yf = np.fft.rfft(np.asarray(y, dtype=np.float64))
    mag = np.abs(Yf.copy())
    mag[0] = -np.inf                       # never count DC among the "keep"
    keep = np.argsort(-mag)[:n_keep]
    out = np.zeros_like(Yf)
    out[0] = Yf[0]
    out[keep] = Yf[keep]
    return np.fft.irfft(out, n=len(y))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def rmse(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def circ_err(phi_hat, phi_true):
    """Smallest absolute phase difference, wrapped to [0, pi]."""
    d = np.mod(np.asarray(phi_hat) - np.asarray(phi_true) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def match_to_truth(comps, f_true):
    """Greedy nearest-frequency matching of estimated components to truth.

    Returns an array (len(f_true), 3); rows are NaN where nothing matched.
    """
    f_true = np.asarray(f_true, dtype=np.float64)
    out = np.full((len(f_true), 3), np.nan)
    if len(comps) == 0:
        return out
    avail = list(range(len(comps)))
    for t, ft in enumerate(f_true):
        if not avail:
            break
        j = min(avail, key=lambda i: abs(comps[i, 0] - ft))
        out[t] = comps[j]
        avail.remove(j)
    return out


# --------------------------------------------------------------------------
# signals defined by the manuscript
# --------------------------------------------------------------------------
def two_tone(N=100, x_max=3.0, f=(0.7, 1.5)):
    """Sec. IV: y = sin(2*pi*f1*x) + sin(2*pi*f2*x), f1=0.7, f2=1.5, N=100, x in [0,3]."""
    x = np.linspace(0.0, x_max, N)
    y = sum(np.sin(2 * np.pi * fi * x) for fi in f)
    return x, y


def multi_tone(N=512, x_max=1.0, f=(5.0, 13.3, 21.7, 30.0), amps=None, phases=None):
    """Sec. V (denoising): y = sum_k A_k sin(2*pi*f_k*x + phi_k)."""
    x = np.linspace(0.0, x_max, N)
    amps = np.ones(len(f)) if amps is None else np.asarray(amps, dtype=np.float64)
    phases = np.zeros(len(f)) if phases is None else np.asarray(phases, dtype=np.float64)
    y = sum(a * np.sin(2 * np.pi * fi * x + p) for a, fi, p in zip(amps, f, phases))
    return x, y
