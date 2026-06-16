#!/usr/bin/env python
"""
anisotropy.py  --  Task 3.8: best-N-term (anisotropy) gap.

At a matched coefficient budget N = (L+1)^2, compares spherical-harmonic DEGREE
truncation (keep all degrees <= L) with the ORACLE best-N-term approximation (keep
the N largest-magnitude coefficients across all degrees).  The gap

    gap = eps_degree(L) - eps_bestN((L+1)^2)   >= 0

measures how much a degree-anisotropic coefficient selection could beat plain
degree truncation.  RESULT: a NULL -- the gap is small (~0.3 percentage points at
the 1% level for the geoid: 0.99% degree-trunc vs 0.68% best-N-term at the same
37,249 coefficients), confirming that SH degree truncation is essentially
best-N-term globally (consistent with the n-width coefficient-optimality
eta_coeff = 1).  This is reported as a confirmatory negative result, NOT a headline.

Writes results/anisotropy.csv.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
import shtrunc as S


def field_rows(field):
    s = field["s"]
    Pl = s.spectrum()
    total = Pl.sum()
    tail = np.cumsum(Pl[::-1])[::-1]            # tail[l] = sum_{k>=l} Pl

    def eps_deg(L):                            # degree-truncation eps at integer L
        t = tail[L + 1] if L + 1 < len(tail) else 0.0
        return float(np.sqrt(t / total))

    Ls = np.arange(len(Pl))
    epsL = np.array([eps_deg(int(L)) for L in Ls])

    rows = []
    for thr in (0.05, 0.01):
        if epsL.min() > thr:                   # never reaches thr within lref
            continue
        L = int(np.argmax(epsL <= thr))        # smallest integer L meeting thr
        N = (L + 1) ** 2
        e_deg = eps_deg(L)
        e_best = float(S.best_n_term_eps(s, [N])[0])
        rows.append(dict(
            field=field["key"], label=field["label"], threshold=thr,
            L_at_threshold=L, n_coeffs=N,
            eps_degree=round(e_deg, 6), eps_best_n_term=round(e_best, 6),
            gap_pp=round(100.0 * (e_deg - e_best), 4) + 0.0,  # +0.0 normalises -0.0
        ))
    return rows


def main():
    print("=" * 78)
    print("Best-N-term (anisotropy) gap (Task 3.8) -- expect a NULL (~0.3pp geoid)")
    print("=" * 78)
    fields = [S.gravity_scalar_field(), S.magnetic_scalar_field()]
    lf = S.lithospheric_scalar_field()
    if lf is not None:
        fields.append(lf)

    rows = []
    for f in fields:
        rows.extend(field_rows(f))
    df = pd.DataFrame(rows)
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    out = os.path.join(C.DIR_RESULTS, "anisotropy.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    g1 = df[(df.field == "gravity") & (df.threshold == 0.01)]
    if len(g1):
        gap = float(g1.iloc[0].gap_pp)
        print("\nGeoid anisotropy gap at 1%%: %.3f percentage points -> %s"
              % (gap, "NULL: SH ~ best-N-term, no meaningful anisotropy gain "
                 "(confirms eta_coeff ~ 1)" if gap < 0.5
                 else "NON-NEGLIGIBLE -- investigate"))
        # TDD assertions: best-N-term is never WORSE than degree truncation (the
        # real mathematical invariant), and the geoid gap is the expected small
        # null (well under half a percentage point).
        assert all(df.gap_pp >= -1e-6), "best-N-term must not be worse than degree trunc"
        assert gap < 0.5, "geoid 1%% anisotropy gap unexpectedly large"
    print("\nWrote %s" % out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
