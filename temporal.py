#!/usr/bin/env python
"""
temporal.py  --  Task 3.3: temporal IGRF (1900-2025) truncation analysis.

For each 5-year epoch the IGRF-13 main field B_r degree variance is synthesised,
the EFFECTIVE reference degree is detected as the last non-zero degree (10 for the
pre-2000 DGRF epochs, which carry no degrees 11-13; 13 from 2000 on), and the
eps(l_max) decay is summarised over a degree window that avoids the empty tail
(fit window [1,9] when l_ref_eff=10, [1,12] when l_ref_eff=13).  Each epoch's eps
is measured against ITS OWN effective reference degree.

Outputs results/temporal_igrf.csv and reports the secular trend.  The clean,
reference-independent flattening comparator is the dipole power fraction
P_1 / sum_l P_l: the axial dipole has weakened over the century, so the relative
non-dipole power has grown and the spectrum has flattened.

NOTE: the pre-1950 DGRF epochs rest on sparse historical observations, so the
decay slope shows a data-quality excursion around 1945-1950 (elevated degree
9-10 power); this is a property of IGRF-13's historical coefficients, not a code
artifact.  The dipole power fraction is smooth across this period and is the more
robust temporal comparator.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import pyshtools.datasets.Earth as Earth

import config as C
import shtrunc as S
from audit import raw_Br_degree_variance      # B_r degree variance from raw Gauss coeffs

YEARS = list(range(1900, 2026, 5))            # 1900, 1905, ..., 2025 (26 epochs)


def epoch_stats(year):
    """One epoch: detect l_ref_eff, fit the eps-curve slope, 1% threshold, dipole
    fraction."""
    igrf = Earth.IGRF_13(lmax=13, year=float(year))
    Pl = raw_Br_degree_variance(igrf, 13)            # index = degree, Pl[0]=0
    nz = np.nonzero(Pl)[0]
    lref_eff = int(nz.max())                          # last non-zero degree (10 or 13)

    lmax_list = np.arange(1, lref_eff + 1)
    eps = S.eps_from_degree_variance(Pl, lmax_list, lmin=1)   # vs this epoch's own l_ref_eff

    fit_range = (1, 9) if lref_eff <= 10 else (1, 12)
    m = (lmax_list >= fit_range[0]) & (lmax_list <= fit_range[1])
    slope = S.fit_loglog(lmax_list[m].astype(float), eps[m])[0]
    spec = S.fit_spectrum(Pl, fit_range)

    l_at = S.crossing_degrees(lmax_list.astype(float), eps, [0.01])[0.01]
    # ceil-then-(l+1)^2 coeff-count convention (project-wide, cf. ROADMAP 0.3)
    coeffs = None if l_at is None else (int(math.ceil(l_at)) + 1) ** 2

    # dipole always dominates the IGRF main field, so Pl[1] > 0 for every epoch
    dipole_frac = float(Pl[1] / Pl[1:].sum())         # P_1 / total power (l>=1)

    return dict(
        year=year,
        lref_eff=lref_eff,
        fit_lo=fit_range[0], fit_hi=fit_range[1],
        decay_slope=round(float(slope), 6),
        spectrum_alpha=round(float(spec["alpha"]), 6),
        dipole_power_frac=round(dipole_frac, 6),
        l_at_1pct=None if l_at is None else round(float(l_at), 3),
        coeffs_at_1pct=coeffs,
    )


def main():
    print("=" * 78)
    print("Temporal IGRF truncation (Task 3.3): 1900-2025, 5-year epochs")
    print("=" * 78)
    rows = [epoch_stats(y) for y in YEARS]
    df = pd.DataFrame(rows)
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    out = os.path.join(C.DIR_RESULTS, "temporal_igrf.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    # Secular trend.  decay_slope has a step at 2000 (the band widens 10->13), so
    # the cleanest cross-century comparator is the dipole power fraction.
    d0, d1 = df.iloc[0], df.iloc[-1]
    frac_trend = np.polyfit(df.year, df.dipole_power_frac, 1)[0]
    print("\nSecular trend:")
    print("  dipole power fraction %.4f (1900) -> %.4f (2025)  [%+.5f / yr -> %s]"
          % (d0.dipole_power_frac, d1.dipole_power_frac, frac_trend,
             "flattening (dipole weakening, relatively more non-dipole power)"
             if frac_trend < 0 else "steepening"))
    post = df[df.year >= 2000]
    if len(post) >= 2:
        sl_tr = np.polyfit(post.year, post.decay_slope, 1)[0]
        print("  decay slope (2000-2025, homogeneous l_ref=13, window [1,12]): "
              "%+.4f -> %+.4f  [%+.5f / yr]"
              % (post.iloc[0].decay_slope, post.iloc[-1].decay_slope, sl_tr))
    print("  (the 1945-1950 slope dip is an IGRF historical-data-quality feature, "
          "not a code artifact; the dipole fraction is smooth across it.)")

    # quick TDD assertions (deterministic; tiny IGRF file, no big download)
    assert all(r["lref_eff"] in (10, 13) for r in rows), "unexpected l_ref_eff"
    assert all(df[df.year < 2000].lref_eff == 10), "pre-2000 should be l_ref_eff=10"
    assert all(df[df.year >= 2000].lref_eff == 13), "2000+ should be l_ref_eff=13"
    assert all(r["decay_slope"] < 0 for r in rows), "slope should be negative"
    print("\nWrote %s" % out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
