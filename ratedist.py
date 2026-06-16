"""
ratedist.py  --  Rate-distortion floor (reverse water-filling) for spherical-
harmonic potential fields.

Component 2 of the Phase 3 novelty core (ROADMAP task 3.5).

Theory (Cover & Thomas, Chap. 10 reverse water-filling)
--------------------------------------------------------
Model each field as a parallel-Gaussian source.  At degree l there are (2l+1)
independent Gaussian coefficients, each with per-coefficient variance

    sigma^2_l = P_l / (2l+1)

where P_l is the field's per-degree variance (degree variance, indexed by
degree, from SHCoeffs.spectrum()).  The total source variance is

    V = sum_l P_l     (sum over the field's non-zero degrees)

Reverse water-filling at a target relative L2 error eps
--------------------------------------------------------
1.  Absolute distortion target:   D_abs = eps^2 * V

2.  Distortion-as-a-function-of-water-level:
        D(mu) = sum_l (2l+1) * min(sigma^2_l, mu)
    D is continuous and strictly increasing from 0 to V on mu in (0, sigma^2_max].

3.  Find mu* via bisection so that D(mu*) = D_abs.

4.  Rate (bits):
        B_WF = sum_l (2l+1) * max(0, 0.5 * log2(sigma^2_l / mu*))

This B_WF is the information-theoretic floor: no lossless-encodable quantiser
can represent the field to within distortion D_abs in fewer bits.

Baseline: float32 SH truncation
--------------------------------
The matched SH-truncation baseline keeps all (ceil(l_trunc)+1)^2 coefficients
and stores each in 32-bit float:

    B_float32 = (ceil(l_trunc) + 1)^2 * 32   bits

where l_trunc is the crossing degree where eps(l) first drops to the target
epsilon threshold (log-log interpolated, as in crossing_degrees).

The ratio B_float32 / B_WF quantifies how many times more bits naive float32
truncation spends compared to the rate-distortion floor.  This ratio is
eps-DEPENDENT (it varies across the 1%/2.5%/5%/10% targets tabulated here).

Public API
----------
    water_fill_spectrum(Pl, eps_targets, lmin=None)  ->  list of dicts
    run_all_fields()                                 ->  nested dict (main entry)

Output
------
    results/water_fill.json   (written by run_all_fields)
"""
from __future__ import annotations

import json
import os

import numpy as np

import config as C
import shtrunc as S


# --------------------------------------------------------------------------- #
# Core: reverse water-filling for a single parallel-Gaussian spectrum
# --------------------------------------------------------------------------- #

def _D_of_mu(mu: float, sigma2: np.ndarray, counts: np.ndarray) -> float:
    """D(mu) = sum_l counts_l * min(sigma^2_l, mu).

    Parameters
    ----------
    mu      : water level (scalar)
    sigma2  : per-coefficient variance, shape (n_degrees,)
    counts  : (2l+1) for each degree, same shape as sigma2
    """
    return float(np.sum(counts * np.minimum(sigma2, mu)))


def _bisect_mu(D_abs: float, sigma2: np.ndarray, counts: np.ndarray,
               tol: float = 1e-14) -> float:
    """Find mu* such that D(mu*) == D_abs via bisection.

    D is continuous and strictly increasing on (0, sigma2.max()].
    We bracket with lo=0, hi=sigma2.max() and iterate until the interval
    width < tol * D_abs (relative stopping criterion).

    Returns mu*.
    """
    lo = 0.0
    hi = float(sigma2.max())

    D_hi = _D_of_mu(hi, sigma2, counts)
    if D_abs >= D_hi:
        # Target distortion >= total variance: any mu >= sigma2.max() achieves
        # D = V.  Return hi so B_WF = 0 (allocate zero bits to every dimension).
        return hi

    if D_abs <= 0.0:
        return 0.0

    # Bisect mu in (0, hi].
    for _ in range(200):                        # 200 iterations is overkill (~60 bits)
        mid = 0.5 * (lo + hi)
        Dmid = _D_of_mu(mid, sigma2, counts)
        if Dmid < D_abs:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < tol * (D_abs + 1e-300):
            break

    return 0.5 * (lo + hi)


def _rate_distortion(Pl: np.ndarray, eps: float, lmin: int) -> dict:
    """Compute the reverse-water-filling rate for a single (Pl, eps) pair.

    Parameters
    ----------
    Pl   : per-degree variance array indexed by degree (Pl[l] = degree-l power).
           Length = lmax+1.  Values at degrees < lmin are expected to be zero
           and are excluded from the computation.
    eps  : target relative L2 error (e.g. 0.01 for 1%)
    lmin : lowest non-zero degree (2 for geoid, 1 for magnetic, 16 for crustal)

    Returns a dict with:
        eps            float  target relative error
        D_abs          float  absolute distortion target = eps^2 * total_var
        mu             float  solved water level
        B_WF_bits      float  rate-distortion floor (bits)
        bits_per_coeff float  B_WF / N_total
        N_total        int    sum of (2l+1) over all non-zero degrees
        total_var      float  sum_l Pl over non-zero degrees
    """
    Pl = np.asarray(Pl, dtype=float)
    # Work only on degrees with Pl > 0, starting from lmin.
    ls_all = np.arange(len(Pl))
    mask = (ls_all >= lmin) & (Pl > 0.0)
    ls = ls_all[mask]
    Pl_nz = Pl[mask]

    counts = (2 * ls + 1).astype(float)         # (2l+1) per degree
    sigma2 = Pl_nz / counts                     # per-coefficient variance

    total_var = float(Pl_nz.sum())
    N_total = int(counts.sum())

    D_abs = float(eps ** 2) * total_var

    mu = _bisect_mu(D_abs, sigma2, counts)

    # Rate: B_WF = sum_l (2l+1) * max(0, 0.5 * log2(sigma2_l / mu))
    with np.errstate(divide="ignore", invalid="ignore"):
        if mu > 0.0:
            log2_ratio = np.log2(sigma2 / mu)          # may be -inf / nan at mu >> sigma2
        else:
            # mu == 0 -> allocate infinite bits; guard for D_abs <= 0 edge case
            log2_ratio = np.full_like(sigma2, np.inf)

    contribution = 0.5 * np.maximum(0.0, log2_ratio)
    B_WF = float(np.sum(counts * contribution))

    bits_per_coeff = B_WF / N_total if N_total > 0 else 0.0

    return dict(
        eps=float(eps),
        D_abs=D_abs,
        mu=float(mu),
        B_WF_bits=B_WF,
        bits_per_coeff=bits_per_coeff,
        N_total=N_total,
        total_var=total_var,
    )


# --------------------------------------------------------------------------- #
# Baseline: float32 SH truncation
# --------------------------------------------------------------------------- #

def _float32_baseline(Pl: np.ndarray, eps: float, lmin: int,
                      lmax_list=None) -> dict:
    """Compute the float32 truncation baseline matched to the same eps.

    Uses crossing_degrees (log-log interpolation) on the Parseval eps-curve
    built from Pl to find l_trunc, then:

        n_coeffs_trunc = (ceil(l_trunc) + 1)^2
        B_float32      = n_coeffs_trunc * 32    (bits)

    Parameters
    ----------
    Pl        : degree variance array (index = degree, length = lmax+1)
    eps       : target relative error
    lmin      : lowest non-zero degree
    lmax_list : optional list of integer degrees to sample; defaults to
                np.arange(lmin, len(Pl))

    Returns a dict with:
        l_trunc        float  interpolated crossing degree (or None)
        n_coeffs_trunc int    (ceil(l_trunc)+1)^2
        B_float32_bits int    n_coeffs_trunc * 32
    """
    Pl = np.asarray(Pl, dtype=float)
    lmax = len(Pl) - 1
    if lmax_list is None:
        lmax_list = list(range(lmin, lmax + 1))
    lmax_arr = np.asarray(lmax_list, dtype=int)

    eps_arr = S.eps_from_degree_variance(Pl, lmax_arr, lmin)
    crossings = S.crossing_degrees(lmax_arr, eps_arr, [eps])
    l_trunc = crossings[eps]                         # float or None

    if l_trunc is None:
        return dict(l_trunc=None, n_coeffs_trunc=None, B_float32_bits=None)

    n_coeffs = (int(np.ceil(l_trunc)) + 1) ** 2
    B_float32 = n_coeffs * 32

    return dict(
        l_trunc=float(l_trunc),
        n_coeffs_trunc=int(n_coeffs),
        B_float32_bits=int(B_float32),
    )


# --------------------------------------------------------------------------- #
# High-level: one (Pl, lmin, lmax_list) + list of eps targets -> list of rows
# --------------------------------------------------------------------------- #

def water_fill_spectrum(Pl: np.ndarray, eps_targets, lmin: int,
                        lmax_list=None) -> list:
    """Rate-distortion floor analysis for a single field spectrum.

    For each eps in eps_targets, computes:
      - reverse water-filling: mu, B_WF_bits, bits_per_coeff
      - float32 SH truncation baseline: l_trunc, n_coeffs_trunc, B_float32_bits
      - ratio: B_float32_bits / B_WF_bits  (savings factor, eps-dependent)

    Parameters
    ----------
    Pl          : degree variance array, shape (lmax+1,); Pl[l] = per-degree power
    eps_targets : iterable of floats in (0, 1] (e.g. [0.10, 0.05, 0.025, 0.01])
    lmin        : lowest non-zero degree (2=geoid, 1=magnetic, 16=crustal)
    lmax_list   : optional list of integer truncation degrees for eps baseline

    Returns
    -------
    list of dicts, one per eps target, each with keys:
        eps, mu, B_WF_bits, bits_per_coeff, l_trunc, n_coeffs_trunc,
        B_float32_bits, float32_over_floor_ratio
    """
    Pl = np.asarray(Pl, dtype=float)
    rows = []
    for eps in eps_targets:
        rd = _rate_distortion(Pl, eps, lmin)
        bl = _float32_baseline(Pl, eps, lmin, lmax_list=lmax_list)

        B_WF = rd["B_WF_bits"]
        B_f32 = bl["B_float32_bits"]

        if B_f32 is not None and B_WF > 0.0:
            ratio = B_f32 / B_WF
        elif B_f32 is not None and B_WF == 0.0:
            ratio = float("inf")
        else:
            ratio = None

        row = dict(
            eps=float(eps),
            mu=rd["mu"],
            B_WF_bits=rd["B_WF_bits"],
            bits_per_coeff=rd["bits_per_coeff"],
            l_trunc=bl["l_trunc"],
            n_coeffs_trunc=bl["n_coeffs_trunc"],
            B_float32_bits=bl["B_float32_bits"],
            float32_over_floor_ratio=ratio,
        )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Per-field spectrum extraction  (reads shtrunc fields, no side effects)
# --------------------------------------------------------------------------- #

def _gravity_spectrum_info():
    """Return (Pl, lmin, lmax_list) for the EGM2008 geoid field."""
    field = S.gravity_scalar_field()
    s = field["s"]
    Pl = s.spectrum()
    lmin = 2
    lmax_list = list(np.arange(lmin, field["lref"] + 1))
    return Pl, lmin, lmax_list, field


def _magnetic_spectrum_info():
    """Return (Pl, lmin, lmax_list) for the IGRF-13 B_r field."""
    field = S.magnetic_scalar_field()
    s = field["s"]
    Pl = s.spectrum()
    lmin = 1
    lmax_list = list(range(1, field["lref"] + 1))
    return Pl, lmin, lmax_list, field


def _crustal_spectrum_info():
    """Return (Pl, lmin, lmax_list) for the NGDC-720 crustal B_r field.
    Returns None if the dataset is unavailable."""
    field = S.lithospheric_scalar_field()
    if field is None:
        return None
    s = field["s"]
    Pl = s.spectrum()
    lmin = 16
    lmax_list = list(np.arange(lmin, field["lref"] + 1))
    return Pl, lmin, lmax_list, field


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

EPS_TARGETS = [0.10, 0.05, 0.025, 0.01]

_FIELD_SPECS = [
    ("gravity",    _gravity_spectrum_info),
    ("magnetic",   _magnetic_spectrum_info),
    ("crustal",    _crustal_spectrum_info),
]


def run_all_fields(eps_targets=None, outpath=None) -> dict:
    """Run the rate-distortion floor analysis for gravity, magnetic, crustal fields.

    Writes results/water_fill.json and prints a readable table.

    Parameters
    ----------
    eps_targets : list of float; defaults to [0.10, 0.05, 0.025, 0.01]
    outpath     : path to write JSON; defaults to C.DIR_RESULTS/water_fill.json

    Returns
    -------
    Nested dict: field_key -> list of result dicts (one per eps)
    """
    if eps_targets is None:
        eps_targets = EPS_TARGETS
    if outpath is None:
        outpath = os.path.join(C.DIR_RESULTS, "water_fill.json")

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    output = {}

    # ----------------------------------------------------------------- table header
    hdr_fmt = "%-12s  %6s  %14s  %14s  %11s  %14s  %10s  %9s"
    row_fmt = "%-12s  %6s  %14.4e  %14.4e  %11.4f  %14.0f  %10.0f  %9.2f"
    print()
    print("=" * 110)
    print("Rate-Distortion Floor (Reverse Water-Filling) vs Float32 SH Truncation")
    print("=" * 110)
    print(hdr_fmt % ("field", "eps", "mu", "B_WF (bits)", "b/coeff",
                     "B_float32 (bits)", "l_trunc", "f32/floor"))
    print("-" * 110)

    for field_key, spec_fn in _FIELD_SPECS:
        info = spec_fn()
        if info is None:
            print("%-12s  [dataset unavailable; skipped]" % field_key)
            output[field_key] = []
            continue

        Pl, lmin, lmax_list, field = info
        rows = water_fill_spectrum(Pl, eps_targets, lmin, lmax_list=lmax_list)
        output[field_key] = rows

        for r in rows:
            eps_pct = "%.1f%%" % (r["eps"] * 100)
            ratio = r["float32_over_floor_ratio"]
            ratio_str = ("%.2f" % ratio) if ratio is not None else "N/A"
            l_trunc = r["l_trunc"]
            n_coeffs = r["n_coeffs_trunc"]
            B_f32 = r["B_float32_bits"]

            if l_trunc is not None:
                print(row_fmt % (
                    field_key, eps_pct,
                    r["mu"], r["B_WF_bits"], r["bits_per_coeff"],
                    B_f32 if B_f32 is not None else float("nan"),
                    l_trunc,
                    ratio if ratio is not None else float("nan"),
                ))
            else:
                print("%-12s  %6s  %14.4e  %14.4e  %11.4f  %14s  %10s  %9s" % (
                    field_key, eps_pct,
                    r["mu"], r["B_WF_bits"], r["bits_per_coeff"],
                    "N/A", "N/A", "N/A",
                ))

    print("=" * 110)
    print()

    # ----------------------------------------------------------------- highlight table
    print("Summary at eps = 1% and 5% (ratios are eps-DEPENDENT, not constant)")
    print("-" * 70)
    for field_key in output:
        rows = output[field_key]
        if not rows:
            continue
        for r in rows:
            if r["eps"] in (0.01, 0.05):
                ratio = r["float32_over_floor_ratio"]
                ratio_str = ("%.2fx" % ratio) if ratio is not None else "N/A"
                print("  %-12s  eps=%5.1f%%  B_WF=%10.0f bits  B_float32=%10s bits  "
                      "float32/floor = %s" % (
                          field_key,
                          r["eps"] * 100,
                          r["B_WF_bits"],
                          ("%d" % r["B_float32_bits"]) if r["B_float32_bits"] else "N/A",
                          ratio_str,
                      ))
    print()

    # ----------------------------------------------------------------- write JSON
    with open(outpath, "w") as fh:
        json.dump(output, fh, indent=2)
    print("Wrote: %s" % outpath)

    return output


# --------------------------------------------------------------------------- #
# Module entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_all_fields()
