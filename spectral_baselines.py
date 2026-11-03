"""
Classical continuous-frequency (parametric) line-spectral estimators, used as
state-of-the-art comparators for SinENR.

Added in revision in response to Referee 1, point 1: a zero-padded FFT is not
the appropriate comparator for estimating the parameters of a small number of
sinusoids.  The estimators below are the standard parametric alternatives and,
unlike the DFT, none of them restricts frequency to a grid.

  * nls_sinusoids   -- nonlinear least squares at known model order K, solved
                       by variable projection (Golub & Pereyra 1973); this is
                       the maximum-likelihood estimator for additive white
                       Gaussian noise (Rife & Boorstyn 1974, Stoica & Moses).
  * music           -- MUSIC pseudospectrum (Schmidt 1986) with forward-
                       backward spatially-smoothed covariance.
  * esprit          -- TLS-ESPRIT on the signal subspace (Roy & Kailath 1989).
  * matrix_pencil   -- Hua & Sarkar (1990) direct-data matrix pencil.

All four assume a uniform sampling grid, which both manuscript benchmarks use.
A real sinusoid is a conjugate pair of complex exponentials, so the subspace
methods are run at model order 2K and the positive-frequency representatives
are returned.

Output convention matches `sinenr.spectral_readout`: an array of rows
(f, phi, a) such that the component is  a * sin(2*pi*f*x + phi)  with a >= 0,
f >= 0 and phi measured at x = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import hankel, lstsq
from scipy.optimize import least_squares


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _design(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """[sin(2 pi f_k x) | cos(2 pi f_k x) | 1] design matrix."""
    th = 2.0 * np.pi * np.outer(x, np.asarray(freqs, dtype=np.float64))
    return np.hstack([np.sin(th), np.cos(th), np.ones((len(x), 1))])


def ls_amp_phase(x, y, freqs) -> np.ndarray:
    """Linear least squares for amplitude and phase at fixed frequencies.

    Returns rows (f, phi, a) with a >= 0, using
    A sin(th) + B cos(th) = hypot(A, B) * sin(th + atan2(B, A)).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    if freqs.size == 0:
        return np.empty((0, 3))
    K = freqs.size
    coef, *_ = lstsq(_design(x, freqs), y)
    A, B = coef[:K], coef[K:2 * K]
    amp = np.hypot(A, B)
    phi = np.arctan2(B, A)
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi
    out = np.stack([freqs, phi, amp], axis=1)
    return out[np.argsort(out[:, 0])]


def reconstruct(comps: np.ndarray, x) -> np.ndarray:
    """Evaluate sum_k a_k sin(2 pi f_k x + phi_k) at arbitrary coordinates."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros_like(x)
    for f, phi, a in np.asarray(comps, dtype=np.float64):
        if np.isfinite(f):
            out += a * np.sin(2.0 * np.pi * f * x + phi)
    return out


def _check_uniform(x: np.ndarray) -> float:
    d = np.diff(x)
    dx = float(np.mean(d))
    if not np.allclose(d, dx, rtol=1e-8, atol=1e-12):
        raise ValueError("subspace estimators require a uniform sampling grid")
    return dx


def _periodogram_init(x, y, K, pad=16) -> np.ndarray:
    """Top-K local maxima of a zero-padded periodogram (the standard NLS start)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N, dx = len(y), float(np.mean(np.diff(x)))
    Y = np.abs(np.fft.rfft(y - y.mean(), n=N * pad))
    fg = np.fft.rfftfreq(N * pad, d=dx)
    cand = [i for i in range(1, len(Y) - 1) if Y[i] >= Y[i - 1] and Y[i] >= Y[i + 1]]
    cand.sort(key=lambda i: -Y[i])
    f0 = np.sort(fg[cand[:K]]) if cand else np.linspace(0, 0.5 / dx, K + 2)[1:-1]
    if len(f0) < K:                         # pad out if too few peaks were found
        f0 = np.concatenate([f0, np.linspace(fg[1], 0.5 / dx, K - len(f0))])
    return np.sort(f0)[:K]


# --------------------------------------------------------------------------
# 1. nonlinear least squares at known model order (variable projection)
# --------------------------------------------------------------------------
def nls_sinusoids(x, y, K, f_init=None, f_max=None, extra_starts=(),
                  n_random_starts=10, seed=0, max_nfev=20000) -> np.ndarray:
    """Separable nonlinear least squares for K real sinusoids plus a constant.

    Only the K frequencies are optimized; amplitudes, phases and the offset are
    eliminated analytically at every step (variable projection).  The search is
    started from the zero-padded periodogram peaks, from any `extra_starts`
    supplied by the caller, and from `n_random_starts` draws from the admissible
    band; the fit with the smallest residual is returned.  The random restarts
    matter on gapped records, where the periodogram start can fall into a
    frequency-collision minimum with large cancelling amplitudes -- the same
    nonconvexity that multi-start training addresses in SinENR.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = float(np.mean(np.diff(x)))
    f_nyq = 0.5 / dx
    hi = f_nyq if f_max is None else min(float(f_max), f_nyq)
    lo = 1.0 / (10.0 * (x.max() - x.min()))

    def residual(f):
        A = _design(x, np.clip(f, lo, hi))
        coef, *_ = lstsq(A, y)
        return y - A @ coef

    starts = [np.asarray(f_init, dtype=np.float64) if f_init is not None
              else _periodogram_init(x, y, K)]
    starts += [np.asarray(s, dtype=np.float64) for s in extra_starts]
    if n_random_starts:
        rng = np.random.default_rng(seed + 7717)
        starts += [np.sort(rng.uniform(lo, hi, K)) for _ in range(int(n_random_starts))]

    best, best_cost = None, np.inf
    for s in starts:
        s = np.sort(np.clip(np.asarray(s, dtype=np.float64).ravel()[:K], lo, hi))
        if s.size < K:
            continue
        try:
            sol = least_squares(residual, s, bounds=(lo, hi), method="trf",
                                xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                max_nfev=max_nfev)
        except Exception:
            continue
        if sol.cost < best_cost:
            best, best_cost = sol.x, sol.cost
    if best is None:
        return np.empty((0, 3))
    return ls_amp_phase(x, y, best)


# --------------------------------------------------------------------------
# subspace machinery shared by MUSIC / ESPRIT
# --------------------------------------------------------------------------
def _data_matrix(y: np.ndarray, m: int) -> np.ndarray:
    """m x (N-m+1) Hankel data matrix whose columns are length-m snapshots."""
    N = len(y)
    return hankel(y[:m], y[m - 1:]) if m <= N else None


def _fb_covariance(y: np.ndarray, m: int) -> np.ndarray:
    """Forward--backward spatially-smoothed covariance of a real record."""
    X = _data_matrix(y, m)
    J = np.fliplr(np.eye(m))
    Rf = X @ X.conj().T
    Rb = J @ Rf.conj() @ J
    return (Rf + Rb) / (2.0 * X.shape[1])


def _default_m(N: int, p: int) -> int:
    """Subarray length: N/2 is the usual choice, floored to keep m > p."""
    return int(max(p + 2, min(N // 2, N - p - 2)))


def music(x, y, K, m=None, n_grid=200001, f_max=None, remove_mean=False) -> np.ndarray:
    """MUSIC frequency estimates, refined by parabolic interpolation.

    The signal subspace has dimension p = 2K because each real sinusoid
    contributes a conjugate pair of complex exponentials.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = _check_uniform(x)
    f_nyq = 0.5 / dx
    hi = f_nyq if f_max is None else min(float(f_max), f_nyq)
    p = 2 * K
    m = _default_m(len(y), p) if m is None else int(m)

    R = _fb_covariance(y - (y.mean() if remove_mean else 0.0), m)
    w, V = np.linalg.eigh(R)
    En = V[:, : m - p]                               # noise subspace

    fg = np.linspace(1.0 / (10.0 * np.ptp(x)), hi, int(n_grid))
    Amat = np.exp(2j * np.pi * np.outer(np.arange(m) * dx, fg))
    proj = np.sum(np.abs(En.conj().T @ Amat) ** 2, axis=0)
    P = 1.0 / np.maximum(proj, 1e-300)

    logP = np.log(P)
    cand = [i for i in range(1, len(P) - 1) if logP[i] >= logP[i - 1] and logP[i] >= logP[i + 1]]
    cand.sort(key=lambda i: -logP[i])
    df = fg[1] - fg[0]
    freqs = []
    for i in cand[:K]:
        a0, a1, a2 = logP[i - 1], logP[i], logP[i + 1]
        den = a0 - 2 * a1 + a2
        d = 0.5 * (a0 - a2) / den if den != 0 else 0.0
        freqs.append(fg[i] + float(np.clip(d, -0.5, 0.5)) * df)
    if not freqs:
        return np.empty((0, 3))
    return ls_amp_phase(x, y, np.sort(freqs))


def esprit(x, y, K, m=None, f_max=None, remove_mean=False) -> np.ndarray:
    """TLS-ESPRIT on the Hankel signal subspace."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = _check_uniform(x)
    f_nyq = 0.5 / dx
    hi = f_nyq if f_max is None else min(float(f_max), f_nyq)
    p = 2 * K
    m = _default_m(len(y), p) if m is None else int(m)

    X = _data_matrix(y - (y.mean() if remove_mean else 0.0), m)
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    Us = U[:, :p]
    U1, U2 = Us[:-1, :], Us[1:, :]

    # total-least-squares solution of U1 Psi = U2
    C = np.hstack([U1, U2])
    _, _, Vh = np.linalg.svd(C.conj().T @ C)
    V = Vh.conj().T
    V12, V22 = V[:p, p:], V[p:, p:]
    try:
        Psi = -V12 @ np.linalg.inv(V22)
    except np.linalg.LinAlgError:
        Psi = np.linalg.pinv(U1) @ U2

    z = np.linalg.eigvals(Psi)
    f = np.angle(z) / (2.0 * np.pi * dx)
    f = np.sort(f[f > 0])
    f = f[(f <= hi)]
    if len(f) == 0:
        return np.empty((0, 3))
    return ls_amp_phase(x, y, f[:K])


def matrix_pencil(x, y, K, L=None, f_max=None, remove_mean=False) -> np.ndarray:
    """Hua & Sarkar matrix-pencil estimator with SVD rank truncation."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = _check_uniform(x)
    f_nyq = 0.5 / dx
    hi = f_nyq if f_max is None else min(float(f_max), f_nyq)
    p = 2 * K
    N = len(y)
    L = int(max(p + 1, min(N // 3, N - p - 1))) if L is None else int(L)

    yc = y - (y.mean() if remove_mean else 0.0)
    Y = hankel(yc[:N - L], yc[N - L - 1:])           # (N-L) x (L+1)
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.conj().T[:, :p]                           # rank-p filtered
    V1, V2 = V[:-1, :], V[1:, :]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    f = np.angle(z) / (2.0 * np.pi * dx)
    f = np.sort(f[f > 0])
    f = f[f <= hi]
    if len(f) == 0:
        return np.empty((0, 3))
    return ls_amp_phase(x, y, f[:K])


# --------------------------------------------------------------------------
# convenience: all four at once
# --------------------------------------------------------------------------
ESTIMATORS = {
    "NLS (known K)": nls_sinusoids,
    "MUSIC": music,
    "ESPRIT": esprit,
    "Matrix pencil": matrix_pencil,
}


def estimate_all(x, y, K, f_max=None, nls_from_subspace=True) -> dict:
    """Run every estimator; NLS is additionally started from the ESPRIT solution."""
    out = {}
    for name, fn in ESTIMATORS.items():
        if name.startswith("NLS"):
            continue
        try:
            out[name] = fn(x, y, K, f_max=f_max)
        except Exception as exc:                      # pragma: no cover
            out[name] = np.empty((0, 3))
            out[name + " error"] = str(exc)
    extra = []
    if nls_from_subspace:
        for nm in ("ESPRIT", "Matrix pencil", "MUSIC"):
            c = out.get(nm)
            if c is not None and len(c) == K:
                extra.append(c[:, 0])
    out["NLS (known K)"] = nls_sinusoids(x, y, K, f_max=f_max, extra_starts=extra)
    return out


if __name__ == "__main__":
    # self-check on the clean two-tone benchmark of the manuscript
    N, x_max, ftrue = 100, 3.0, np.array([0.7, 1.5])
    xs = np.linspace(0, x_max, N)
    ys = np.sin(2 * np.pi * ftrue[0] * xs) + np.sin(2 * np.pi * ftrue[1] * xs)
    for nm, c in estimate_all(xs, ys, K=2).items():
        if not isinstance(c, np.ndarray):
            print(nm, c)
            continue
        print(f"{nm:16s} f = {np.array2string(c[:, 0], precision=9)}  "
              f"|df| = {np.mean(np.abs(c[:, 0] - ftrue)):.3e}  "
              f"|dphi| = {np.mean(np.abs(c[:, 1])):.3e}  "
              f"|da| = {np.mean(np.abs(c[:, 2] - 1)):.3e}")
