"""
codec.py  --  Operational codec gap analysis for spherical-harmonic fields.

Component 3 of the Phase 3 novelty core (ROADMAP task 3.6).

Concept (Appendix A C3)
-----------------------
The rate-distortion (water-filling) floor B_WF from ratedist.py is an
information-theoretic lower bound on the bits needed to represent a field to
relative L2 error eps.  It assumes:
  - continuous-valued coefficients (no integer rounding of bit allocations),
  - optimal Gaussian codes (ideal entropy coding matched to the source).

A real operational codec sits ABOVE the floor because:
  1. Bit allocations must be NON-NEGATIVE INTEGERS (integer rounding gap).
  2. Coefficients are quantized with a finite-resolution scalar quantizer
     (uniform quantizer per degree).
  3. An entropy coder can approach but not beat the empirical entropy of the
     quantized symbols.

This module builds and evaluates that operational codec and reports
eps-DEPENDENT ratios comparing it to the floor and to naive float32 storage.

Low-degree / short-band caveats (visible in the magnetic IGRF-13 results)
-------------------------------------------------------------------------
For a band-limited field with few degrees (IGRF-13 has only l=1..13, so at most
2l+1=27 coefficients per degree), two honest artefacts appear and are EXPECTED,
not bugs:
  - B_entropy can fall BELOW B_WF.  The empirical Shannon entropy of a handful of
    quantized symbols per degree is a downward-biased (finite-sample / Miller-Madow)
    estimate of the true Gaussian source entropy, so the entropy-coded estimate can
    dip under the asymptotic floor.  B_entropy is therefore an OPTIMISTIC estimate
    for short bands; B_real (the integer-bit rate) is the conservative figure.
  - achieved_eps can sit well below the target (e.g. magnetic at eps=5% achieves
    ~3.4%).  The minimum non-zero integer allocation at the dominant l=1 degree
    over-allocates bits, so the codec OVERSHOOTS accuracy (conservative).  This is
    why the magnetic greedy top-up runs zero steps.

Integer bit allocation
----------------------
Given the water-fill per-degree allocation b_l (real-valued, from the reverse
water-fill solution), the integer allocation is:

    b_l_int = max(0, ceil(b_l))

where ceil() is the ceiling function (rounds up to the next integer).  This
choice is necessary to guarantee the invariant B_real >= B_WF: since
ceil(x) >= x for all non-negative x, the integer rate sum
    B_real = sum_l (2l+1) * ceil(b_l) >= sum_l (2l+1) * b_l = B_WF.
The ROADMAP (Appendix A C3) explicitly permits either round() or ceil(),
noting "pick one, document it."  We choose ceil() to ensure the operational
rate sits at or above the information-theoretic floor by construction.  The
downside -- that ceil over-allocates slightly more than round -- is acceptable
and is what produces the observable eta_bits = B_real / B_WF ratio reported
in the codec gap table.

Uniform quantizer per degree
-----------------------------
For degree l with integer allocation b = b_l_int > 0, a mid-tread uniform
quantizer is applied independently to each of the (2l+1) coefficients at that
degree.  The quantizer step is:

    Delta_l = max(|c_lm|) / (2^(b-1))     for b >= 1

where the max is taken over the (2l+1) coefficients at degree l.  This gives
the full range [-max|c|, max|c|] split into 2^b bins, matching the coefficient
magnitudes.  The quantization is:

    c_lm_q = Delta_l * round(c_lm / Delta_l)

clipped to the range [-max|c|, max|c|] to avoid overflow.  Degrees with b=0
are not quantized (their coefficients are dropped; they contribute zero
reconstructed power and their full power goes into distortion).

Greedy top-up
-------------
Integer rounding can occasionally increase the total distortion above the
target D_abs = eps^2 * total_var.  If this occurs, a greedy top-up loop bumps
the bit allocation at the degree with the HIGHEST remaining distortion
contribution by +1 until the distortion target is satisfied.  Each greedy step
reduces distortion by re-quantizing that degree at (b+1) bits.

Rates reported
--------------
B_real      = sum_l (2l+1) * b_l_int          (uniform-quantizer rate, bits)
B_entropy   = sum_l (2l+1) * H_l              (entropy-coded rate, bits)
              where H_l = Shannon entropy (bits) of the quantized symbols at l
B_float32   = (ceil(l_trunc)+1)^2 * 32        (naive float32 SH truncation)
B_WF        = rate-distortion floor (from ratedist._rate_distortion)

eta_bits    = B_real / B_WF                   (operational gap; expected ~1-2x)
float32_saving = B_float32 / B_entropy        (improvement over naive storage)
achieved_eps = sqrt(D_quant / total_var)      (confirm <= target eps)

Public API
----------
    codec_gap(Pl, coeffs_by_degree, eps_targets, lmin)  ->  list of dicts
    run_all_fields()                                     ->  nested dict (main)

Output
------
    results/codec_gap.json   (written by run_all_fields)
"""
from __future__ import annotations

import json
import os

import numpy as np

import config as C
import shtrunc as S
from ratedist import _rate_distortion, _float32_baseline, water_fill_spectrum


# --------------------------------------------------------------------------- #
# Helpers: per-degree water-fill real-valued bit allocation
# --------------------------------------------------------------------------- #

def _water_fill_bits_per_degree(Pl: np.ndarray, eps: float,
                                lmin: int) -> tuple:
    """Compute the real-valued water-fill bit allocation per degree.

    Reuses ratedist._rate_distortion to obtain mu, then computes b_l for
    every degree l >= lmin with Pl > 0.

    Returns
    -------
    ls      : ndarray of degree indices (subset of 0..lmax where Pl > 0 & l>=lmin)
    sigma2  : per-coefficient variance Pl/(2l+1), shape (n_degrees,)
    b_l     : real-valued water-fill allocation max(0, 0.5*log2(sigma2/mu)), same shape
    mu      : solved water level
    D_abs   : absolute distortion target = eps^2 * sum_l>=lmin Pl
    total_var : total variance sum_l>=lmin Pl
    """
    Pl = np.asarray(Pl, dtype=float)
    ls_all = np.arange(len(Pl))
    mask = (ls_all >= lmin) & (Pl > 0.0)
    ls = ls_all[mask]
    Pl_nz = Pl[mask]
    counts = (2 * ls + 1).astype(float)
    sigma2 = Pl_nz / counts

    rd = _rate_distortion(Pl, eps, lmin)
    mu = rd["mu"]
    total_var = rd["total_var"]
    D_abs = rd["D_abs"]

    if mu > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            b_l = 0.5 * np.maximum(0.0, np.log2(sigma2 / mu))
    else:
        # mu == 0 (D_abs <= 0): allocate as many bits as the data require
        b_l = np.full_like(sigma2, 0.0)

    return ls, sigma2, b_l, mu, D_abs, total_var


# --------------------------------------------------------------------------- #
# Helpers: uniform quantizer and entropy estimation
# --------------------------------------------------------------------------- #

def _quantize_degree(coeffs: np.ndarray, b: int) -> np.ndarray:
    """Apply a mid-tread uniform quantizer at b bits to a 1-D coefficient array.

    For b == 0 the coefficients are not retained (return zero array).
    For b >= 1 the step size Delta = max(|c|) / 2^(b-1) covers
    [-max|c|, max|c|] with 2^b uniform bins.

    Parameters
    ----------
    coeffs : 1-D array of real coefficients at one degree (all (2l+1) of them)
    b      : integer bit allocation >= 0

    Returns
    -------
    c_q : quantized coefficient array, same shape as coeffs
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if b == 0:
        return np.zeros_like(coeffs)
    max_abs = float(np.max(np.abs(coeffs)))
    if max_abs == 0.0:
        return np.zeros_like(coeffs)
    Delta = max_abs / float(2 ** (b - 1))
    c_q = Delta * np.round(coeffs / Delta)
    # Clip to [-max_abs, max_abs] to prevent any overflows at extreme rounding
    c_q = np.clip(c_q, -max_abs, max_abs)
    return c_q


def _entropy_bits(symbols: np.ndarray) -> float:
    """Shannon entropy (bits) of a discrete 1-D symbol array.

    Computes H = -sum_i p_i * log2(p_i) where p_i are the empirical
    probabilities of each unique quantized level.  Returns 0.0 if all
    symbols are identical (entropy = 0).

    Parameters
    ----------
    symbols : 1-D array; the quantized coefficient values (floats, treated
              as symbols by equality)

    Returns
    -------
    H : float, Shannon entropy in bits per symbol
    """
    symbols = np.asarray(symbols).ravel()
    if len(symbols) == 0:
        return 0.0
    _, counts = np.unique(symbols, return_counts=True)
    n = float(len(symbols))
    probs = counts / n
    # Avoid log2(0); only positive probabilities contribute
    nonzero = probs > 0.0
    H = -float(np.sum(probs[nonzero] * np.log2(probs[nonzero])))
    return H


# --------------------------------------------------------------------------- #
# Core: build and measure the operational codec for one (Pl, coeffs, eps) triple
# --------------------------------------------------------------------------- #

def _operational_codec_one(Pl: np.ndarray,
                           coeffs_by_degree: dict,
                           eps: float,
                           lmin: int) -> dict:
    """Operational integer-quantizer codec for one (spectrum, eps) pair.

    Parameters
    ----------
    Pl               : per-degree variance array, shape (lmax+1,)
    coeffs_by_degree : dict {l: array_of_shape_(2l+1,)}, actual field
                       coefficients at each non-zero degree; l keys must be
                       a superset of the degrees in Pl with Pl>0 and l>=lmin
    eps              : target relative L2 error
    lmin             : lowest non-zero degree

    Returns
    -------
    dict with keys:
        eps, D_abs, total_var, mu
        B_WF (float, bits), B_real (float, bits), B_entropy (float, bits)
        b_l_int_by_degree  (dict l -> int, for inspection)
        achieved_eps (float), D_quant (float)
        eta_bits (B_real / B_WF)
        greedy_topup_steps (int, 0 if no top-up was needed)
    """
    Pl = np.asarray(Pl, dtype=float)
    ls, sigma2, b_l, mu, D_abs, total_var = _water_fill_bits_per_degree(
        Pl, eps, lmin)

    # ---- integer ceiling (guarantees B_real >= B_WF by construction) -------
    b_l_int = np.maximum(0, np.ceil(b_l).astype(int))
    b_l_int_dict = {int(l): int(b) for l, b in zip(ls, b_l_int)}
    counts = (2 * ls + 1).astype(float)

    # ---- quantize and measure initial distortion --------------------------
    D_quant, D_per_degree, c_q_by_degree = _quantize_all_degrees(
        ls, Pl, b_l_int_dict, coeffs_by_degree, lmin)

    # ---- greedy top-up if distortion target is exceeded -------------------
    greedy_steps = 0
    while D_quant > D_abs * (1.0 + 1e-10):   # small tolerance for fp precision
        # Find degree with highest distortion contribution
        worst_idx = int(np.argmax(D_per_degree))
        worst_l = int(ls[worst_idx])
        b_l_int_dict[worst_l] = b_l_int_dict.get(worst_l, 0) + 1

        # Re-quantize only that degree
        if worst_l in coeffs_by_degree:
            c_orig = coeffs_by_degree[worst_l]
            c_q = _quantize_degree(c_orig, b_l_int_dict[worst_l])
            c_q_by_degree[worst_l] = c_q
            diff = c_orig - c_q
            D_per_degree[worst_idx] = float(np.sum(diff ** 2))
            D_quant = float(np.sum(D_per_degree))

        greedy_steps += 1
        if greedy_steps > 10 * len(ls):      # safety guard (should never hit)
            break

    # ---- compute B_real ---------------------------------------------------
    B_real = 0.0
    for l_idx, l in enumerate(ls):
        l = int(l)
        b = b_l_int_dict.get(l, 0)
        n_coeffs_at_l = int(2 * l + 1)
        B_real += n_coeffs_at_l * b

    # ---- compute B_entropy ------------------------------------------------
    B_entropy = 0.0
    for l in ls:
        l = int(l)
        b = b_l_int_dict.get(l, 0)
        n_coeffs_at_l = int(2 * l + 1)
        if b == 0:
            # All coefficients mapped to zero -> one symbol, entropy = 0
            B_entropy += 0.0
        elif l in c_q_by_degree:
            H_l = _entropy_bits(c_q_by_degree[l])
            B_entropy += n_coeffs_at_l * H_l
        else:
            # Fallback: no higher than uniform-quantizer rate
            B_entropy += n_coeffs_at_l * min(b, float(b))

    # ---- achieved relative error ------------------------------------------
    achieved_eps = float(np.sqrt(D_quant / total_var)) if total_var > 0 else 0.0

    # ---- B_WF from the rate-distortion floor ------------------------------
    rd = _rate_distortion(Pl, eps, lmin)
    B_WF = rd["B_WF_bits"]

    eta_bits = (B_real / B_WF) if B_WF > 0.0 else float("inf")

    return dict(
        eps=float(eps),
        D_abs=float(D_abs),
        total_var=float(total_var),
        mu=float(mu),
        B_WF=float(B_WF),
        B_real=float(B_real),
        B_entropy=float(B_entropy),
        b_l_int_by_degree=b_l_int_dict,
        achieved_eps=float(achieved_eps),
        D_quant=float(D_quant),
        eta_bits=float(eta_bits),
        greedy_topup_steps=int(greedy_steps),
    )


def _quantize_all_degrees(ls, Pl, b_l_int_dict, coeffs_by_degree, lmin):
    """Quantize every active degree; return (D_quant, D_per_degree, c_q_dict).

    Degrees with b=0 contribute their full Pl as distortion (the degree is
    dropped entirely).  Degrees with b>0 contribute the sum of squared
    quantization errors.

    D_per_degree is a mutable numpy array (one entry per degree in ls), used
    by the greedy top-up loop.
    """
    n = len(ls)
    D_per_degree = np.zeros(n, dtype=float)
    c_q_by_degree = {}

    for i, l in enumerate(ls):
        l = int(l)
        b = b_l_int_dict.get(l, 0)
        if b == 0:
            # Dropped: all power becomes distortion
            D_per_degree[i] = float(Pl[l])
        elif l in coeffs_by_degree:
            c_orig = np.asarray(coeffs_by_degree[l], dtype=float)
            c_q = _quantize_degree(c_orig, b)
            c_q_by_degree[l] = c_q
            diff = c_orig - c_q
            D_per_degree[i] = float(np.sum(diff ** 2))
        else:
            # Coefficients not provided for this degree; treat as dropped
            D_per_degree[i] = float(Pl[l])

    D_quant = float(np.sum(D_per_degree))
    return D_quant, D_per_degree, c_q_by_degree


# --------------------------------------------------------------------------- #
# Public API: run codec gap analysis for a given field spectrum + coefficients
# --------------------------------------------------------------------------- #

def codec_gap(Pl: np.ndarray, coeffs_by_degree: dict, eps_targets,
              lmin: int, lmax_list=None) -> list:
    """Codec gap analysis for a single field.

    For each eps in eps_targets, computes:
      - B_WF   : rate-distortion floor (ratedist reverse water-fill)
      - B_real : operational uniform-integer-quantizer rate
      - B_entropy : entropy-coded estimate (empirical Shannon entropy)
      - B_float32 : naive float32 SH truncation baseline
      - eta_bits  : B_real / B_WF  (operational gap above the floor)
      - float32_saving : B_float32 / B_entropy
      - achieved_eps    : confirmed <= target eps (after any greedy top-up)

    Parameters
    ----------
    Pl               : degree variance array, shape (lmax+1,); Pl[l] = power at l
    coeffs_by_degree : dict {l: 1-D array of (2l+1) coefficients}
    eps_targets      : iterable of floats in (0, 1]
    lmin             : lowest non-zero degree
    lmax_list        : optional list of integer degrees for float32 baseline

    Returns
    -------
    list of dicts, one per eps, with keys:
        eps, B_WF, B_real, B_entropy, B_float32,
        eta_bits, float32_saving, achieved_eps,
        greedy_topup_steps, D_abs, D_quant, mu
    """
    Pl = np.asarray(Pl, dtype=float)
    rows = []
    for eps in eps_targets:
        rd_result = _operational_codec_one(Pl, coeffs_by_degree, eps, lmin)
        bl_result = _float32_baseline(Pl, eps, lmin, lmax_list=lmax_list)

        B_float32 = bl_result["B_float32_bits"]
        B_entropy = rd_result["B_entropy"]

        if B_float32 is not None and B_entropy > 0.0:
            float32_saving = float(B_float32) / B_entropy
        elif B_float32 is not None and B_entropy == 0.0:
            float32_saving = float("inf")
        else:
            float32_saving = None

        row = dict(
            eps=rd_result["eps"],
            mu=rd_result["mu"],
            B_WF=rd_result["B_WF"],
            B_real=rd_result["B_real"],
            B_entropy=rd_result["B_entropy"],
            B_float32=B_float32,
            l_trunc=bl_result["l_trunc"],
            n_coeffs_trunc=bl_result["n_coeffs_trunc"],
            eta_bits=rd_result["eta_bits"],
            float32_saving=float32_saving,
            achieved_eps=rd_result["achieved_eps"],
            D_abs=rd_result["D_abs"],
            D_quant=rd_result["D_quant"],
            total_var=rd_result["total_var"],
            greedy_topup_steps=rd_result["greedy_topup_steps"],
        )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Field coefficient extraction helpers
# --------------------------------------------------------------------------- #

def _coeffs_by_degree_from_shcoeffs(s, lmin: int, lmax: int) -> dict:
    """Extract actual SH coefficients as a per-degree dict from SHCoeffs.

    For each degree l in [lmin, lmax], collects all (2l+1) real coefficients:
      cos terms: s.coeffs[0, l, 0..l]  (order m=0..l)
      sin terms: s.coeffs[1, l, 1..l]  (order m=1..l, excluding monopole)

    Parameters
    ----------
    s    : pyshtools SHCoeffs (4pi-normalised real)
    lmin : minimum degree (inclusive)
    lmax : maximum degree (inclusive)

    Returns
    -------
    dict {l: 1-D float array of length (2l+1)}
    """
    arr = s.to_array()   # shape (2, lmax+1, lmax+1): [0]=cos, [1]=sin
    coeffs = {}
    for l in range(lmin, lmax + 1):
        cos_coeffs = arr[0, l, :l + 1]             # m = 0..l  (l+1 values)
        sin_coeffs = arr[1, l, 1:l + 1]            # m = 1..l  (l values)
        all_c = np.concatenate([cos_coeffs, sin_coeffs])   # (2l+1) total
        coeffs[l] = all_c
    return coeffs


def _gravity_codec_info():
    """Return (Pl, coeffs_by_degree, lmin, lmax_list) for the gravity field."""
    field = S.gravity_scalar_field()
    s = field["s"]
    Pl = s.spectrum()
    lmin = 2
    lref = field["lref"]
    lmax_list = list(np.arange(lmin, lref + 1))
    coeffs = _coeffs_by_degree_from_shcoeffs(s, lmin, lref)
    return Pl, coeffs, lmin, lmax_list, field


def _magnetic_codec_info():
    """Return (Pl, coeffs_by_degree, lmin, lmax_list) for the magnetic field."""
    field = S.magnetic_scalar_field()
    s = field["s"]
    Pl = s.spectrum()
    lmin = 1
    lref = field["lref"]
    lmax_list = list(range(1, lref + 1))
    coeffs = _coeffs_by_degree_from_shcoeffs(s, lmin, lref)
    return Pl, coeffs, lmin, lmax_list, field


def _crustal_codec_info():
    """Return (Pl, coeffs_by_degree, lmin, lmax_list) for the crustal field.
    Returns None if the dataset is unavailable."""
    field = S.lithospheric_scalar_field()
    if field is None:
        return None
    s = field["s"]
    Pl = s.spectrum()
    lmin = 16
    lref = field["lref"]
    lmax_list = list(np.arange(lmin, lref + 1))
    coeffs = _coeffs_by_degree_from_shcoeffs(s, lmin, lref)
    return Pl, coeffs, lmin, lmax_list, field


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #

EPS_TARGETS = [0.10, 0.05, 0.025, 0.01]

_FIELD_SPECS = [
    ("gravity",  _gravity_codec_info),
    ("magnetic", _magnetic_codec_info),
    ("crustal",  _crustal_codec_info),
]


def run_all_fields(eps_targets=None, outpath=None) -> dict:
    """Run the codec gap analysis for gravity, magnetic, and crustal fields.

    Writes results/codec_gap.json and prints a readable table.

    Parameters
    ----------
    eps_targets : list of float; defaults to [0.10, 0.05, 0.025, 0.01]
    outpath     : path to write JSON; defaults to C.DIR_RESULTS/codec_gap.json

    Returns
    -------
    Nested dict: field_key -> list of result dicts (one per eps)
    """
    if eps_targets is None:
        eps_targets = EPS_TARGETS
    if outpath is None:
        outpath = os.path.join(C.DIR_RESULTS, "codec_gap.json")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    output = {}

    # ----------------------------------------------------------------- table
    col_w = 115
    hdr_fmt = ("%-12s  %6s  %12s  %12s  %12s  %12s  %9s  %9s  %12s  %5s")
    row_fmt = ("%-12s  %6s  %12.0f  %12.0f  %12.0f  %12s  %9.3f  %9.2f  %12.6f  %5d")
    print()
    print("=" * col_w)
    print("Operational Codec Gap: B_WF / B_real / B_entropy / B_float32")
    print("=" * col_w)
    print(hdr_fmt % ("field", "eps", "B_WF(bits)", "B_real(bits)", "B_ent(bits)",
                     "B_f32(bits)", "eta_bits", "f32/ent", "ach_eps", "topup"))
    print("-" * col_w)

    for field_key, info_fn in _FIELD_SPECS:
        info = info_fn()
        if info is None:
            print("%-12s  [dataset unavailable; skipped]" % field_key)
            output[field_key] = []
            continue

        Pl, coeffs, lmin, lmax_list, field = info
        rows = codec_gap(Pl, coeffs, eps_targets, lmin, lmax_list=lmax_list)
        output[field_key] = rows

        for r in rows:
            eps_pct = "%.1f%%" % (r["eps"] * 100)
            B_f32_str = ("%d" % r["B_float32"]) if r["B_float32"] is not None else "N/A"
            fs_str = ("%.2f" % r["float32_saving"]) if r["float32_saving"] is not None else "N/A"
            print(row_fmt % (
                field_key, eps_pct,
                r["B_WF"], r["B_real"], r["B_entropy"],
                B_f32_str,
                r["eta_bits"],
                r["float32_saving"] if r["float32_saving"] is not None else float("nan"),
                r["achieved_eps"],
                r["greedy_topup_steps"],
            ))

    print("=" * col_w)
    print()

    # -------------------------------------------------------- highlight table
    print("Summary at eps = 1% and 5%  (all ratios are eps-DEPENDENT, not constant)")
    print("-" * 90)
    for field_key in output:
        rows = output[field_key]
        if not rows:
            continue
        for r in rows:
            if r["eps"] in (0.01, 0.05):
                fs = r["float32_saving"]
                print(
                    "  %-12s  eps=%5.1f%%  B_WF=%9.0f  B_real=%9.0f  "
                    "B_entropy=%9.0f  B_float32=%9s  "
                    "eta_bits=%.3f  f32/ent=%s  ach_eps=%.6f"
                    % (
                        field_key, r["eps"] * 100,
                        r["B_WF"], r["B_real"], r["B_entropy"],
                        ("%d" % r["B_float32"]) if r["B_float32"] else "N/A",
                        r["eta_bits"],
                        ("%.2fx" % fs) if fs is not None else "N/A",
                        r["achieved_eps"],
                    )
                )
    print()

    # --------------------------------------------------------- write JSON
    with open(outpath, "w") as fh:
        json.dump(output, fh, indent=2)
    print("Wrote: %s" % outpath)

    return output


# --------------------------------------------------------------------------- #
# Module entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_all_fields()
