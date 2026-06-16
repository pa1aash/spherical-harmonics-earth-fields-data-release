#!/usr/bin/env python
"""
planetary.py  --  Task 3.1: planetary geoid truncation (Earth / Moon / Mars).

Builds the geoid (selenoid / areoid) undulation for the Moon (GRGM1200B) and
Mars (GMM3) with the SAME first-order-Bruns pipeline as the Earth geoid, runs the
exact-Parseval eps(l_max) analysis, and writes results/planetary.csv with one row
per body (Earth geoid included as the reference).  The decay slope is the
scale-free comparator across bodies: compressibility tracks each body's spectral
character / source depth.

Usage (pinned conda env):
    python planetary.py

First run downloads GRGM1200B (Moon) and GMM3 (Mars) via pyshtools (cached after).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

import config as C
import shtrunc as S


def _row(body, field, res):
    fit = res["fit"]
    l_at = res["thresholds"][0.01]
    coeffs = None if l_at is None else (int(math.ceil(l_at)) + 1) ** 2
    return dict(
        body=body,
        label=field["label"],
        accessor=field.get("accessor", "Earth.EGM2008"),
        ellipsoid=field.get("ellipsoid", "boule.WGS84"),
        lref=field["lref"],
        decay_slope=round(float(fit["slope"]), 6),
        slope_jackknife_se=round(float(fit["slope_se"]), 6),
        spectrum_alpha=round(float(fit["alpha"]), 6),
        l_at_1pct=None if l_at is None else round(float(l_at), 2),
        coeffs_at_1pct=coeffs,
        parseval_pass=bool(res["parseval"]["passed"]),
        monotonic=bool(res["monotonic"]),
    )


def build():
    fields = [
        ("earth", S.gravity_scalar_field()),
        ("moon", S.planetary_geoid_field("moon")),
        ("mars", S.planetary_geoid_field("mars")),
    ]
    rows = []
    for body, field in fields:
        res = S.analyse_field(field)
        rows.append(_row(body, field, res))
        # quick TDD assertions (the analysis is exact + deterministic):
        assert res["monotonic"], "%s eps not monotonic" % body
        assert res["parseval"]["passed"], "%s Parseval check failed" % body
        assert rows[-1]["decay_slope"] < 0, "%s geoid slope should be negative" % body
    return rows


def write_curves(outdir=None):
    """Emit results/planetary_curves.csv: the EXACT per-degree Parseval error
    eps(L) = sqrt( sum_{l>L} sigma_l^2 / sum_l sigma_l^2 ) at every integer
    degree L for Earth / Moon / Mars, the data source for the planetary
    decay-curve figure (Figure D).  Uses the SAME first-order-Bruns geoid fields
    as build(); no new download (models are cached after the first run)."""
    import numpy as np
    if outdir is None:
        outdir = C.DIR_RESULTS
    fields = [
        ("earth", S.gravity_scalar_field()),
        ("moon", S.planetary_geoid_field("moon")),
        ("mars", S.planetary_geoid_field("mars")),
    ]
    rows = []
    for body, field in fields:
        Pl = field["s"].spectrum()                 # sigma_l^2, index = degree
        lref = field["lref"]
        total = float(Pl.sum())
        for L in range(lref + 1):
            eps = float(np.sqrt(Pl[L + 1:].sum() / total))
            rows.append(dict(body=body, label=field["label"], lref=lref,
                             lmax=L, eps=eps))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "planetary_curves.csv")
    pd.DataFrame(rows, columns=["body", "label", "lref", "lmax", "eps"]).to_csv(
        out, index=False)
    print("Wrote %s (%d rows)" % (out, len(rows)))
    return out


def main():
    print("=" * 72)
    print("Planetary geoid truncation (Task 3.1): Earth / Moon / Mars")
    print("=" * 72)
    rows = build()
    df = pd.DataFrame(rows, columns=[
        "body", "label", "accessor", "ellipsoid", "lref", "decay_slope",
        "slope_jackknife_se", "spectrum_alpha", "l_at_1pct", "coeffs_at_1pct",
        "parseval_pass", "monotonic"])
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    out = os.path.join(C.DIR_RESULTS, "planetary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("\nWrote %s" % out)
    write_curves(C.DIR_RESULTS)
    print("\nDecay slope is the scale-free comparator across bodies "
          "(more negative = more compressible / redder spectrum).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
