"""
Islet glucose--hormone model: numpy port of the authors' Fortran code
(islet_module.f90 / islet_trace.f90).

Each of nmax islets carries three Stuart--Landau-type amplitude--phase
oscillators (alpha, beta, delta cells) coupled within the islet by kmat and
coupled to each other globally through the plasma glucose g:

    d(amp_i)/dt   = (lam_i - |amp_i|^2) amp_i
                    + sum_j kmat[j,i] amp_j cos(theta_j - theta_i)
    d(theta_i)/dt = om_i + mu_i cos(theta_i)
                    + sum_j kmat[j,i] amp_j sin(theta_j - theta_i)/amp_i
    dg/dt         = lamda [ (1-glures) H_alpha g0 - (1-inres) H_beta g ]

with glucose-dependent growth rates lam and phase forcing mu, integrated by
the same fixed-step RK4 and sampled every `isample` steps.

TWO DELIBERATE DEPARTURES from the Fortran, both documented:
  1. The Numerical-Recipes `ran1`/`gasdev` generators seeded from the system
     clock (`ran_seed`) are replaced by numpy's PCG64 with an explicit seed,
     so that a run is reproducible.  A trajectory from this port is therefore
     NOT bit-identical to a Fortran run; the model, the parameters, the
     integrator and the sampling are.
  2. A burn-in interval may be discarded, which the Fortran driver does not do.

Parameter values are those hard-coded in islet_trace.f90.
"""
from __future__ import annotations

import numpy as np

# --- parameters exactly as set in islet_trace.f90 ---------------------
DT, ITMAX, NMAX, ISAMPLE = 0.01, 10_000, 200, 5
G0, EPS, GAM, LAMDA, LAMDMAX, FSLOP, DELG = 7.0, 0.4, 0.1, 1.0, 0.5, 5.0, 2.0
GLURES, INRES = 0.0, 0.0

# kmat as assigned in the driver; the sums use kmat[j, i], i.e. its transpose
KMAT = np.array([[0.0,  EPS,  EPS],
                 [-EPS, 0.0,  EPS],
                 [-EPS, -EPS, 0.0]])
C = KMAT.T.copy()                      # C[i, j] == KMAT[j, i]


def _derivs(g, amp, theta):
    lam = np.array([
        0.5 * (1.0 + np.tanh((-g + G0) / FSLOP)),
        0.5 * (1.0 + np.tanh((g - G0) / FSLOP)),
        LAMDMAX * 0.5 * (1.0 + np.tanh((g - G0 + DELG) / FSLOP)),
    ])
    mu = np.array([GAM * (g - G0), GAM * (G0 - g), GAM * (G0 - g)])

    # pairwise phase differences: d[i, j, n] = theta[j, n] - theta[i, n]
    d = theta[None, :, :] - theta[:, None, :]
    aj = amp[None, :, :]                                   # amp[j, n]
    cij = C[:, :, None]                                    # C[i, j]
    rtmp1 = np.einsum("ijn->in", cij * aj * np.cos(d))
    safe = np.where(np.abs(amp) > 0, amp, np.inf)          # mirrors amp/=0 guard
    rtmp2 = np.einsum("ijn->in", cij * aj * np.sin(d)) / safe

    damp = (lam[:, None] - np.abs(amp) ** 2) * amp + rtmp1
    dtheta = OM + mu[:, None] * np.cos(theta) + rtmp2

    ha = np.sum(amp[0] * (np.cos(theta[0]) + 1.0))
    hb = np.sum(amp[1] * (np.cos(theta[1]) + 1.0))
    dg = LAMDA * ((1.0 - GLURES) * ha * G0 - (1.0 - INRES) * hb * g)
    return damp, dtheta, dg


def _rk4(g, amp, theta):
    da1, dt1, dg1 = _derivs(g, amp, theta)
    k1a, k1p, k1g = DT * da1, DT * dt1, DT * dg1
    da2, dt2, dg2 = _derivs(g + 0.5 * k1g, amp + 0.5 * k1a, theta + 0.5 * k1p)
    k2a, k2p, k2g = DT * da2, DT * dt2, DT * dg2
    da3, dt3, dg3 = _derivs(g + 0.5 * k2g, amp + 0.5 * k2a, theta + 0.5 * k2p)
    k3a, k3p, k3g = DT * da3, DT * dt3, DT * dg3
    da4, dt4, dg4 = _derivs(g + k3g, amp + k3a, theta + k3p)
    k4a, k4p, k4g = DT * da4, DT * dt4, DT * dg4
    return (amp + (k1a + 2 * (k2a + k3a) + k4a) / 6.0,
            theta + (k1p + 2 * (k2p + k3p) + k4p) / 6.0,
            g + (k1g + 2 * (k2g + k3g) + k4g) / 6.0)


OM = None   # set by simulate()


def simulate(seed=0, itmax=ITMAX, nmax=NMAX, burn_in=0.0):
    """Return (t, g_centred, h) with h[0..2] the alpha/beta/delta hormone sums.

    burn_in : fraction of the record to discard before centring.
    """
    global OM
    rng = np.random.default_rng(seed)
    OM = 1.0 + 0.1 * rng.standard_normal((3, nmax))
    g = G0 + 0.1 * rng.standard_normal()
    amp = rng.random((3, nmax))
    theta = 2.0 * np.pi * rng.random((3, nmax))

    dmax = itmax // ISAMPLE
    gdata = np.empty(dmax)
    hdata = np.empty((3, dmax))
    icount = 0
    for it in range(1, itmax + 1):
        neg = amp <= 0.0                       # sign convention of the Fortran
        amp = np.where(neg, -amp, amp)
        theta = np.where(neg, theta + np.pi, theta)
        amp, theta, g = _rk4(g, amp, theta)
        if it % ISAMPLE == 0:
            gdata[icount] = g
            for i in range(3):
                hdata[i, icount] = np.sum(amp[i] * (1.0 + np.cos(theta[i])))
            icount += 1

    t = (np.arange(1, dmax + 1)) * DT * ISAMPLE
    k = int(round(burn_in * dmax))
    t, gdata, hdata = t[k:], gdata[k:], hdata[:, k:]
    return t, gdata - gdata.mean(), hdata


if __name__ == "__main__":
    t, g, h = simulate(seed=0, burn_in=0.25)
    print(f"samples {len(t)}  t in [{t[0]:.2f}, {t[-1]:.2f}]  "
          f"g range [{g.min():.3f}, {g.max():.3f}]  std {g.std():.3f}")
