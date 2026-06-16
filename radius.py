"""
radius.py  --  Component 5: radius dependence of magnetic-field compressibility.

Physical background
-------------------
The IGRF-13 main field is an internal potential field.  Continuing it from
Earth's surface (a = 6371.2 km) downward toward the source scales each
degree-l Gauss coefficient by (a/r)^(l+1).  Because that factor grows with l,
downward continuation AMPLIFIES high degrees, un-reddening the spectrum.  The
field therefore becomes LESS compressible at depth: the surface-most-compressible
field is least-compressible at its source in the outer core.

B_r continuation note
---------------------
The exact downward-continuation operator for B_r is (a/r)^(l+2) on each
degree-l Gauss coefficient (one extra power of a/r relative to the potential).
However, the extra SINGLE power of (a/r) is degree-INDEPENDENT and therefore
cancels exactly in the RELATIVE L2 error eps (it scales numerator and
denominator identically).  Using (a/r)^(l+1) on the Gauss coefficients here
gives an identical eps-curve and l@1% to using (a/r)^(l+2); we adopt (a/r)^(l+1)
to be explicit that the quantity driving compressibility is the l-dependent
part of the continuation factor.

Output
------
results/radius_compressibility.csv : r_km, a_over_r, l_at_1pct, coeffs_at_1pct,
                                     decay_slope
"""
from __future__ import annotations

import os
import csv

import numpy as np
import pyshtools.datasets.Earth as Earth

import config as C
import shtrunc as S

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
A_KM = 6371.2            # IGRF reference radius (same as igrf.r0/1e3)
CMB_KM = 3480.0          # core-mantle boundary radius (km)
LMAX = C.MAG_LREF        # 13
LMIN = 1                 # B_r has no degree-0 monopole
THRESH_1PCT = 0.01

# Fit range for the decay slope (same as the surface magnetic fit range)
FIT_RANGE = C.MAG_FIT_RANGE   # (1, 12)


# --------------------------------------------------------------------------- #
# Core helpers (pure-numpy, no pyshtools datasets -- importable without I/O)
# --------------------------------------------------------------------------- #
def scale_gauss_array(arr, a, r, lmax):
    """Return a copy of arr with each degree l scaled by (a/r)^(l+1).

    Parameters
    ----------
    arr : ndarray, shape (2, lmax+1, lmax+1)
        Schmidt-normalised Gauss coefficients [g (index 0), h (index 1)].
    a : float
        Reference radius (same units as r).
    r : float
        Evaluation radius.  r <= a for downward continuation.
    lmax : int
        Maximum degree.

    Returns
    -------
    ndarray, same shape as arr, with arr[:, l, :] *= (a/r)^(l+1).
    """
    factor = np.array([(a / r) ** (l + 1) for l in range(lmax + 1)])
    # factor shape: (lmax+1,) -- broadcast over (2, lmax+1, lmax+1)
    return arr * factor[np.newaxis, :, np.newaxis]


def Br_degree_variance_from_array(arr, lmax):
    """4pi B_r degree variance from a Schmidt Gauss coefficient array.

    sigma_l^2(B_r) = (l+1)^2 / (2l+1)  *  sum_m (g_lm^2 + h_lm^2)   [nT^2]

    Parameters
    ----------
    arr : ndarray, shape (2, lmax+1, lmax+1)
    lmax : int

    Returns
    -------
    Pl : ndarray, shape (lmax+1,), Pl[0] = 0 (no magnetic monopole).
    """
    ls = np.arange(lmax + 1, dtype=float)
    gh2 = np.sum(arr[0] ** 2 + arr[1] ** 2, axis=1)   # sum over m per degree
    with np.errstate(divide="ignore", invalid="ignore"):
        Pl = (ls + 1) ** 2 / (2 * ls + 1) * gh2
    Pl[0] = 0.0
    return Pl


def radius_sweep(arr_surface, a, r_values, lmax, lmax_list, fit_range, thresh):
    """Compute compressibility metrics across a radius sweep.

    Parameters
    ----------
    arr_surface : ndarray, shape (2, lmax+1, lmax+1)
        Surface-level Schmidt Gauss coefficients.
    a : float
        Reference radius (km).
    r_values : sequence of float
        Radii at which to evaluate (km).
    lmax : int
        Maximum degree of the model.
    lmax_list : sequence of int
        Truncation degrees for the eps-curve.
    fit_range : tuple (lo, hi)
        Inclusive degree range for the log-log slope fit.
    thresh : float
        Single error threshold for l@thresh reporting.

    Returns
    -------
    list of dict, one per r, with keys:
        r_km, a_over_r, Pl, eps, l_at_thresh, coeffs_at_thresh, decay_slope
    """
    lmax_arr = np.asarray(list(lmax_list), dtype=int)
    lo, hi = int(fit_range[0]), int(fit_range[1])

    rows = []
    for r in r_values:
        arr_r = scale_gauss_array(arr_surface, a, r, lmax)
        Pl = Br_degree_variance_from_array(arr_r, lmax)
        eps = S.eps_from_degree_variance(Pl, lmax_arr, lmin=LMIN)

        crossings = S.crossing_degrees(lmax_arr.astype(float), eps, [thresh])
        l_cross = crossings[thresh]   # float or None

        if l_cross is not None:
            n_coeffs = int((int(np.ceil(l_cross)) + 1) ** 2)
        else:
            n_coeffs = None

        mask = (lmax_arr >= lo) & (lmax_arr <= hi) & (eps > 0)
        slope, _, _ = S.fit_loglog(lmax_arr[mask].astype(float), eps[mask])

        rows.append(dict(
            r_km=float(r),
            a_over_r=float(a / r),
            Pl=Pl,
            eps=eps,
            l_at_thresh=l_cross,
            coeffs_at_thresh=n_coeffs,
            decay_slope=float(slope) if slope is not None else float("nan"),
        ))
    return rows


# --------------------------------------------------------------------------- #
# Main analysis (uses the IGRF-13 dataset)
# --------------------------------------------------------------------------- #
def run():
    """Full radius-dependence analysis.

    Loads IGRF-13, constructs the Schmidt array once at the surface, then sweeps
    downward to the CMB.  Writes results/radius_compressibility.csv and prints
    a summary table.

    Returns
    -------
    list of dict (same as radius_sweep output, without the Pl/eps arrays).
    """
    # ------------------------------------------------------------------ #
    # Load IGRF-13 and extract Schmidt (g, h) arrays                      #
    # ------------------------------------------------------------------ #
    igrf = Earth.IGRF_13(lmax=LMAX, year=C.MAG_YEAR)
    a = igrf.r0 / 1e3      # should equal A_KM = 6371.2 km
    assert abs(a - A_KM) < 0.1, "Unexpected IGRF r0: %.3f km" % a

    sch = igrf.convert(normalization="schmidt", csphase=1)
    arr_surface = sch.to_array()   # shape (2, LMAX+1, LMAX+1)

    # ------------------------------------------------------------------ #
    # Radius sweep: 13 log-spaced points, surface -> CMB                  #
    # (geomspace(a, CMB, 13) already includes both endpoints exactly).    #
    # ------------------------------------------------------------------ #
    # Log-spaced between surface and CMB gives adequate resolution.
    r_inner = np.geomspace(a, CMB_KM, 13)          # 13 pts incl. both ends
    r_values = np.unique(np.concatenate([[a], r_inner, [CMB_KM]]))
    r_values = np.sort(r_values)[::-1]              # descending: surface first

    lmax_list = list(range(1, LMAX + 1))            # 1..13

    rows = radius_sweep(arr_surface, a, r_values, LMAX, lmax_list,
                        FIT_RANGE, THRESH_1PCT)

    # ------------------------------------------------------------------ #
    # Write CSV                                                           #
    # ------------------------------------------------------------------ #
    os.makedirs(C.DIR_RESULTS, exist_ok=True)
    csv_path = os.path.join(C.DIR_RESULTS, "radius_compressibility.csv")
    fieldnames = ["r_km", "a_over_r", "l_at_1pct", "coeffs_at_1pct",
                  "decay_slope"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "r_km":          "%.2f" % row["r_km"],
                "a_over_r":      "%.6f" % row["a_over_r"],
                "l_at_1pct":     ("%.3f" % row["l_at_thresh"]
                                  if row["l_at_thresh"] is not None else ""),
                "coeffs_at_1pct": (str(row["coeffs_at_thresh"])
                                   if row["coeffs_at_thresh"] is not None else ""),
                "decay_slope":   "%.4f" % row["decay_slope"],
            })
    print("Wrote %s" % csv_path)

    # ------------------------------------------------------------------ #
    # Print summary table                                                 #
    # ------------------------------------------------------------------ #
    _print_table(rows, a)
    return [{k: v for k, v in r.items() if k not in ("Pl", "eps")}
            for r in rows]


def _print_table(rows, a):
    hdr = ("%-10s  %-8s  %-10s  %-14s  %-12s" %
           ("r_km", "a/r", "l@1%", "coeffs@1%", "decay_slope"))
    sep = "-" * len(hdr)
    print()
    print("=== Radius-Dependence of Magnetic-Field Compressibility (IGRF-13) ===")
    print(hdr)
    print(sep)
    for row in rows:
        l_str = ("%.2f" % row["l_at_thresh"]
                 if row["l_at_thresh"] is not None else "  n/a ")
        c_str = (str(row["coeffs_at_thresh"])
                 if row["coeffs_at_thresh"] is not None else "  n/a ")
        tag = ""
        if abs(row["r_km"] - a) < 0.1:
            tag = " <- surface"
        elif abs(row["r_km"] - CMB_KM) < 1.0:
            tag = " <- CMB"
        print("%-10.2f  %-8.4f  %-10s  %-14s  %-12.4f%s" %
              (row["r_km"], row["a_over_r"], l_str, c_str,
               row["decay_slope"], tag))
    print(sep)
    surface_row = rows[0]
    cmb_row = rows[-1]
    if surface_row["l_at_thresh"] is not None and cmb_row["l_at_thresh"] is not None:
        print("l@1%% reversal:  surface %.2f  →  CMB %.2f  (ΔCompressibility: %s)"
              % (surface_row["l_at_thresh"], cmb_row["l_at_thresh"],
                 "LESS compressible at depth (l@1%% increased)"
                 if cmb_row["l_at_thresh"] > surface_row["l_at_thresh"]
                 else "unchanged/more compressible at depth"))
    else:
        print("l@1%% not bracketed at CMB (spectrum too flat to reach 1%% error "
              "within 13 degrees -- consistent with maximum un-reddening).")
    print()


if __name__ == "__main__":
    run()
