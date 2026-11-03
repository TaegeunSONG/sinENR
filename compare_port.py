"""
Is the numpy port of the islet model statistically equivalent to the Fortran?

The manuscript states that a trajectory from `islet_model.py` is statistically
equivalent to, but not bit-identical with, one from the original Fortran. That
claim rested on the port being a faithful transcription; this script measures
it.

The two implementations cannot agree pointwise. They draw their random numbers
from different generators (xorshift64 in the released Fortran, PCG64 in the
port) and in a different order, so each run is a different realization of the
same ensemble. What is testable is the distribution of trajectory statistics
over independent runs, which is what is compared here.

Procedure
  * Compile and run `islet_trace.f90`, which already emits `itrymax`
    independent trajectories of the centred plasma glucose.
  * Run `islet_model.simulate` the same number of times with seeds 0, 1, ...
  * Process both identically -- discard the first 25 % as transient, take
    every 4th sample, centre -- which is the record the paper analyses.
  * For each trajectory compute the standard deviation and the dominant
    periodogram frequency, and compare the two samples with a two-sided
    Kolmogorov-Smirnov test.

Note that the Fortran seeds itself from the wall clock, so its side of the
comparison is not reproducible run to run; the Python side is seeded.

Usage:  python compare_port.py [n_reps]
        -> results/port_comparison.csv  and a console summary
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
import islet_model as IM

BURN_IN, STRIDE = 0.25, 4
FBAND = (0.05, 0.5)          # where the islet rhythm lives


def _metrics(t, g):
    """Summary statistics of one trajectory.

    The grid-level dominant frequency `f_bin` is retained for the record, but
    it is not a useful statistic for a distributional test: on a record of
    this length it can take only a handful of values (the DFT grid spacing is
    1/L = 0.0134), so the two samples are compared on continuous quantities
    instead -- the parabolically interpolated peak and the amplitude-weighted
    spectral centroid of the band.
    """
    g = g - g.mean()
    dx = float(np.mean(np.diff(t)))
    mag = np.abs(np.fft.rfft(g))
    fg = np.fft.rfftfreq(len(g), d=dx)
    sel = (fg >= FBAND[0]) & (fg <= FBAND[1])
    fs, ms = fg[sel], mag[sel]

    k = int(np.argmax(ms))
    f_bin = float(fs[k])

    # parabolic interpolation on the log magnitude, as in sinenr.fft_peaks
    if 0 < k < len(ms) - 1:
        a0, a1, a2 = (np.log(ms[k - 1] + 1e-300), np.log(ms[k] + 1e-300),
                      np.log(ms[k + 1] + 1e-300))
        den = a0 - 2 * a1 + a2
        d = 0.5 * (a0 - a2) / den if den != 0 else 0.0
        f_peak = f_bin + float(np.clip(d, -0.5, 0.5)) * (fs[1] - fs[0])
    else:
        f_peak = f_bin

    f_centroid = float(np.average(fs, weights=ms))
    return float(g.std()), f_bin, f_peak, f_centroid


def run_fortran(n_reps):
    """Compile islet_trace.f90 and parse its trajectories."""
    gf = None
    for cand in ("gfortran", "/opt/homebrew/bin/gfortran", "/usr/local/bin/gfortran"):
        if subprocess.run(["which", cand], capture_output=True).returncode == 0 \
           or os.path.exists(cand):
            gf = cand
            break
    if gf is None:
        raise RuntimeError("gfortran not found; cannot run the Fortran side")

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(HERE, "islet_trace.f90")
        # islet_trace.f90 hard-codes itrymax=100; patch only the replicate count
        text = open(src).read().replace("itrymax=100", f"itrymax={n_reps}")
        patched = os.path.join(td, "islet_trace.f90")
        open(patched, "w").write(text)
        exe = os.path.join(td, "islet_trace")
        r = subprocess.run([gf, "-std=legacy", "-O2", "-o", exe, patched],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("gfortran failed:\n" + r.stderr[-2000:])
        r = subprocess.run([exe], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("islet_trace failed:\n" + r.stderr[-2000:])

    runs, cur = [], []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            cur.append((float(parts[0]), float(parts[1])))
        elif cur:
            runs.append(np.array(cur))
            cur = []
    if cur:
        runs.append(np.array(cur))

    out = []
    for a in runs[:n_reps]:
        t, g = a[:, 0], a[:, 1]
        k = int(round(BURN_IN * len(t)))
        out.append(_metrics(t[k::STRIDE], g[k::STRIDE]))
    return out


def run_python(n_reps):
    out = []
    for seed in range(n_reps):
        t, g, _ = IM.simulate(seed=seed, burn_in=BURN_IN)
        out.append(_metrics(t[::STRIDE], g[::STRIDE]))
    return out


def main():
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs(OUT, exist_ok=True)

    print(f"Fortran: compiling and running {n_reps} trajectories ...", flush=True)
    f = run_fortran(n_reps)
    print(f"Python : running {n_reps} trajectories ...", flush=True)
    p = run_python(n_reps)

    cols = ("sd", "f_bin", "f_peak", "f_centroid")
    rows = ([dict(source="fortran", **dict(zip(cols, m))) for m in f]
            + [dict(source="python", **dict(zip(cols, m))) for m in p])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "port_comparison.csv"), index=False)

    print(f"\n{'quantity':26s} {'Fortran':>22s} {'Python':>22s} {'KS p':>8s}")
    for col, label in [("sd", "trajectory s.d."),
                       ("f_peak", "peak freq (interpolated)"),
                       ("f_centroid", "spectral centroid")]:
        a = df[df.source == "fortran"][col].to_numpy()
        b = df[df.source == "python"][col].to_numpy()
        ks = ks_2samp(a, b)
        print(f"{label:26s} {a.mean():10.4f} +- {a.std():7.4f} "
              f"{b.mean():10.4f} +- {b.std():7.4f} {ks.pvalue:8.3f}")

    print("\nGrid-level dominant bin (not tested; only a few values are "
          "attainable at 1/L = %.4f):" % (1.0 / 74.8))
    tab = (df.assign(f_bin=df.f_bin.round(5))
             .pivot_table(index="f_bin", columns="source", values="sd",
                          aggfunc="count", fill_value=0))
    print(tab.to_string())
    print(f"\nn = {n_reps} trajectories per implementation; "
          f"wrote results/port_comparison.csv")


if __name__ == "__main__":
    main()
