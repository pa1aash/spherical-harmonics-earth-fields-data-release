#!/usr/bin/env python
"""
resolution.py  --  Task 3.9: spherical-harmonic resolution table.

Maps truncation degree l_max to the spatial half-wavelength it resolves and the
number of coefficients it stores:

    half-wavelength  lambda/2 = pi R / l_max     (R = mean Earth radius 6371 km)
    coefficients     N = (l_max + 1)^2

This makes explicit that degree 2190 (the EGM2008 maximum) corresponds to a
~9.1 km harmonic half-wavelength, NOT the ~200 m figure sometimes quoted for such
models -- that ~200 m is the spacing of the auxiliary topographic grid used to
augment the harmonic model (Hirt et al. 2013), not the harmonic resolution.

Writes results/resolution.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C

R_KM = 6371.0                          # mean Earth radius
DEGREES = [10, 13, 36, 90, 180, 360, 720, 1080, 2190]


def main():
    rows = []
    for lmax in DEGREES:
        rows.append(dict(
            degree=lmax,
            half_wavelength_km=round(float(np.pi * R_KM / lmax), 2),
            n_coeffs=(lmax + 1) ** 2,
        ))
    df = pd.DataFrame(rows)
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    out = os.path.join(C.DIR_RESULTS, "resolution.csv")
    df.to_csv(out, index=False)
    print("Spherical-harmonic resolution table (lambda/2 = pi R / l, R=6371 km)")
    print(df.to_string(index=False))
    # sanity: degree 2190 is ~9.1 km, NOT 200 m; coeffs = 4,800,481
    d2190 = df[df.degree == 2190].iloc[0]
    assert abs(d2190.half_wavelength_km - 9.14) < 0.1, "degree-2190 half-wavelength wrong"
    assert int(d2190.n_coeffs) == 4800481, "degree-2190 coeff count wrong"
    print("\nWrote %s" % out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
