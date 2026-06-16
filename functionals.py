#!/usr/bin/env python
"""
functionals.py  --  Additional gravity functional truncation analysis.

Computes, for each of four EGM2008-derived gravity functionals (geoid,
disturbance, anomaly, Trr), the Parseval-based eps(l_max) curve and
summary statistics, then writes results/functionals.csv.

Each functional is a DIAGONAL per-degree rescaling of the disturbing-potential
degree variance (exact, grid-free).  See shtrunc.gravity_functional_field for
the weight convention (anomaly uses w_l = (l-1), NOT (l+1)).

Usage:
    python functionals.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import config as C
import shtrunc as S

_KINDS = ["geoid", "disturbance", "anomaly", "Trr"]
_LMAX_LIST = np.asarray(sorted(set(C.GRAV_LMAX_LIST)), dtype=int)
_FIT_RANGE = C.GRAV_FIT_RANGE          # config-authoritative (4, 300)
_LMIN = 2
_THRESHOLD_1PCT = 0.01
# All functional slopes use the raw-Stokes disturbing-potential Parseval path
# (NOT the DH-grid round-trip): the 'geoid' functional here is ~-0.898, vs the
# Table I round-tripped headline ~-0.861 (the gap is the Bruns/normal-gravity
# factor).  This 'method' tag is written to the CSV so the artifact is self-
# documenting and cannot be misread as contradicting Table I.
_METHOD = "raw_stokes_disturbing_potential_parseval"


def compute_functional_stats(kind):
    """Compute summary statistics for one gravity functional.

    Returns a dict with: kind, label, decay_slope, spectrum_alpha,
    l_at_1pct, coeffs_at_1pct, lref.
    """
    fd = S.gravity_functional_field(kind)
    Pl = fd["Pl"]
    lref = fd["lref"]

    # eps curve over the gravity degree grid
    eps = S.eps_from_degree_variance(Pl, _LMAX_LIST, lmin=_LMIN)

    # eps-OLS decay slope over fit_range
    fit_mask = (_LMAX_LIST >= _FIT_RANGE[0]) & (_LMAX_LIST <= _FIT_RANGE[1])
    decay_slope, _, _ = S.fit_loglog(_LMAX_LIST[fit_mask].astype(float),
                                     eps[fit_mask])

    # spectral index alpha from degree-variance fit
    spec = S.fit_spectrum(Pl, _FIT_RANGE)
    spectrum_alpha = spec["alpha"]

    # 1% crossing degree + coefficients (ceil convention matching make_figures)
    crossings = S.crossing_degrees(_LMAX_LIST.astype(float), eps, [_THRESHOLD_1PCT])
    l_at = crossings[_THRESHOLD_1PCT]
    if l_at is not None:
        coeffs_at_1pct = (int(math.ceil(l_at)) + 1) ** 2
        l_at_1pct = round(l_at, 2)
    else:
        l_at_1pct = None
        coeffs_at_1pct = None

    return dict(
        functional=kind,
        label=fd["label"],
        decay_slope=round(float(decay_slope), 6),
        spectrum_alpha=round(float(spectrum_alpha), 6),
        l_at_1pct=l_at_1pct,
        coeffs_at_1pct=coeffs_at_1pct,
        lref=lref,
        method=_METHOD,
    )


def write_functionals():
    """Compute all four functionals and write results/functionals.csv."""
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    rows = [compute_functional_stats(k) for k in _KINDS]
    df = pd.DataFrame(rows, columns=["functional", "label", "decay_slope",
                                     "spectrum_alpha", "l_at_1pct",
                                     "coeffs_at_1pct", "lref", "method"])
    out_path = os.path.join(C.DIR_RESULTS, "functionals.csv")
    df.to_csv(out_path, index=False)
    print("Wrote %s" % out_path)
    return df


def main():
    print("=" * 72)
    print("Gravity functional truncation analysis (Task 3.2)")
    print("=" * 72)
    df = write_functionals()
    print(df.to_string(index=False))
    print()
    print("Decay slopes (eps-OLS, fit range l=%d..%d):" % _FIT_RANGE)
    for _, row in df.iterrows():
        print("  %-14s  decay_slope=%+.4f  alpha=%.4f  l@1%%=%s  coeffs=%s"
              % (row["functional"], row["decay_slope"], row["spectrum_alpha"],
                 row["l_at_1pct"], row["coeffs_at_1pct"]))
    print()
    print("Higher-order functionals should decay SLOWER (less negative slope).")
    slopes = df.set_index("functional")["decay_slope"].to_dict()
    order_ok = (slopes["Trr"] > slopes["disturbance"] > slopes["geoid"])
    print("  slope ordering (Trr > disturbance > geoid): %s"
          % ("PASS" if order_ok else "FAIL"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
