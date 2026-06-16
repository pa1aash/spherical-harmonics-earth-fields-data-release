"""
bandlimit.py  --  Effective spectral bandlimit computation (GATE C / Component 1).

Computes the degree l_useful beyond which truncation stores estimation NOISE
rather than signal, derived from each model's own published error variances.

Two crossover metrics per field:
    Per-degree crossover:
        l_useful_perdegree = smallest l >= l_min where S_l < N_l  (SNR_l < 1)

    Cumulative-tail crossover:
        l_useful_cumulative = smallest l where
            tail_signal(l) = sum_{k > l} S_k  <  tail_noise(l) = sum_{k > l} N_k

For GRAVITY (EGM2008, l_nominal=2190):
    S_l from g.spectrum()[0]  (calibrated-error 'geoid' power spectrum)
    N_l from g.spectrum()[1]  (calibrated-error spectrum in same units)
    Cross-check: N_l vs np.sum(g.errors[0]**2 + g.errors[1]**2, axis=1) * r0^2
    (the r0^2 factor comes from the 'geoid' convention in g.spectrum())

For MAGNETIC (IGRF-13, l_nominal=13):
    Base field epoch 2020.0, later epoch 2025.0.
    S_l = per-degree power of the base field (Lowes-Mauersberger spectrum).
    N_l = per-degree power of the 5-year SV: sum_m (dC_lm^2 + dS_lm^2) per degree,
          where dC/dS are the 4pi-normalised Gauss coefficient differences
          (m25.to_array() - m20.to_array()).

Results written to:
    results/bandlimit.csv  (field, l_useful_perdegree, l_useful_cumulative,
                            l_nominal, snr_below_10_deg, snr_below_2_deg,
                            noise_proxy, notes)

All heavy computation (EGM2008 lmax=2190) uses g.spectrum() which returns
pre-allocated per-degree arrays; no large intermediate coefficient arrays are
built for the crossover computation itself.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pyshtools.datasets.Earth as Earth

import config as C


# --------------------------------------------------------------------------- #
# 1.  Crossover helpers — operate on 1-D arrays S_l and N_l (indexed by degree)
# --------------------------------------------------------------------------- #

def _perdegree_crossover(S, N, l_min):
    """Return the smallest integer degree >= l_min where S[l] < N[l].

    Parameters
    ----------
    S, N : array-like, shape (L+1,)
        Signal and noise degree-variance arrays, indexed by degree.
    l_min : int
        First degree to consider (degrees 0/1 are excluded for geoid; 1 for B_r).

    Returns
    -------
    int or None
        First degree where S[l] < N[l], or None if the crossover is not found
        within the array.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    lmax = len(S) - 1
    for l in range(l_min, lmax + 1):
        if S[l] < N[l]:
            return l
    return None


def _cumulative_tail_crossover(S, N, l_min):
    """Return the smallest integer degree >= l_min where tail_S(l) < tail_N(l).

    tail_signal(l) = sum_{k > l} S[k]
    tail_noise(l)  = sum_{k > l} N[k]

    Parameters
    ----------
    S, N : array-like, shape (L+1,)
        Signal and noise degree-variance arrays.
    l_min : int
        First degree to consider.

    Returns
    -------
    int or None
        First degree where the signal tail drops below the noise tail, or None.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    lmax = len(S) - 1
    # tail[l] = sum_{k > l} X[k] = cumsum from the right, shifted by 1
    # tail[lmax] = 0, tail[lmax-1] = X[lmax], etc.
    S_tail = np.concatenate([np.cumsum(S[::-1])[::-1][1:], [0.0]])
    N_tail = np.concatenate([np.cumsum(N[::-1])[::-1][1:], [0.0]])
    for l in range(l_min, lmax + 1):
        if S_tail[l] < N_tail[l]:
            return l
    return None


def _first_below_snr(S, N, threshold, l_min):
    """Return the smallest degree >= l_min where SNR_l = S[l]/N[l] < threshold.

    Returns None if no such degree exists in the array or if N[l] == 0.
    """
    S = np.asarray(S, dtype=float)
    N = np.asarray(N, dtype=float)
    lmax = len(S) - 1
    for l in range(l_min, lmax + 1):
        if N[l] > 0 and S[l] / N[l] < threshold:
            return l
    return None


# --------------------------------------------------------------------------- #
# 2.  Gravity (EGM2008 full model, lmax=2190)
# --------------------------------------------------------------------------- #

def compute_gravity_bandlimit(verbose=True):
    """Load EGM2008 with calibrated errors (lmax=2190) and compute the effective
    spectral bandlimit.

    Returns a dict with all derived quantities.  Uses g.spectrum() which
    returns per-degree arrays; no large intermediate coefficient arrays are
    built.
    """
    if verbose:
        print("Loading EGM2008 lmax=2190 with calibrated errors (may take a "
              "few minutes, cache is at ~/.cache/pyshtools/)...")

    g = Earth.EGM2008(lmax=2190)

    # g.spectrum() returns (signal_spectrum, error_spectrum) as a tuple when
    # errors are loaded.  Both arrays have length lmax+1 = 2191, indexed by
    # degree.  We use function='geoid' (default), unit='per_l' (default), so
    # each entry is the total degree-l contribution to the geoid power in m^2.
    sp = g.spectrum()                   # tuple: (S_array, N_array) of shape (2191,)
    assert isinstance(sp, tuple) and len(sp) == 2, (
        "Expected g.spectrum() to return a 2-tuple (signal, error); got %r" % type(sp))

    S = sp[0]   # per-degree signal power (geoid, m^2), shape (2191,)
    N = sp[1]   # per-degree noise power (calibrated errors, geoid, m^2)

    assert len(S) == 2191, "Expected 2191 degrees (0..2190); got %d" % len(S)
    assert len(N) == 2191, "Expected 2191 error degrees; got %d" % len(N)

    # --- Cross-check: N_l vs raw error array sum ----------------------------
    # g.errors has shape (2, lmax+1, lmax+1) in 4pi normalisation.
    # g.spectrum(function='geoid') scales by r0^2, so per-degree error power =
    #     r0^2 * sum_m (errors[0,l,m]^2 + errors[1,l,m]^2)
    r0 = g.r0                           # reference radius (m)
    N_raw = np.sum(g.errors[0] ** 2 + g.errors[1] ** 2, axis=1) * r0 ** 2
    # Degrees 0..2190: N_raw is shape (2191,)
    # The cross-check should match N to numerical precision.
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.where(N > 0, np.abs(N - N_raw) / N, np.abs(N - N_raw))
    max_rel_diff = float(np.max(rel_diff))

    if verbose:
        print("  Cross-check max rel diff (N from spectrum vs N from raw errors): "
              "%.4e" % max_rel_diff)

    # --- SNR table at selected degrees -------------------------------------
    snr_probe_degrees = [100, 360, 720, 1000, 1500, 2000, 2160]
    snr_table = {}
    for ld in snr_probe_degrees:
        if ld <= 2190 and N[ld] > 0:
            snr_table[ld] = float(S[ld] / N[ld])
        elif ld <= 2190:
            snr_table[ld] = float("inf")
        else:
            snr_table[ld] = None

    # --- Degree where SNR first drops below 10 and below 2 -----------------
    l_min = 2                           # geoid: degrees 0,1 excluded
    snr_below_10 = _first_below_snr(S, N, threshold=10.0, l_min=l_min)
    snr_below_2  = _first_below_snr(S, N, threshold=2.0,  l_min=l_min)

    # --- Per-degree crossover (SNR_l -> 1) ----------------------------------
    l_useful_pd = _perdegree_crossover(S, N, l_min=l_min)

    # --- Cumulative-tail crossover ------------------------------------------
    l_useful_ct = _cumulative_tail_crossover(S, N, l_min=l_min)

    if verbose:
        print("\n--- GRAVITY (EGM2008) EFFECTIVE BANDLIMIT ---")
        print("  l_nominal = 2190")
        print("  l_useful (per-degree SNR < 1) : %s" % l_useful_pd)
        print("  l_useful (cumulative tail)     : %s" % l_useful_ct)
        print("  SNR first < 10  at degree      : %s" % snr_below_10)
        print("  SNR first <  2  at degree      : %s" % snr_below_2)
        print("\n  Per-degree SNR_l at selected degrees:")
        print("  %8s  %14s  %14s  %12s" % ("degree", "S_l (m^2)", "N_l (m^2)", "SNR_l"))
        print("  " + "-" * 54)
        for ld in snr_probe_degrees:
            if snr_table[ld] is not None:
                sl = float(S[ld]) if ld <= 2190 else float("nan")
                nl = float(N[ld]) if ld <= 2190 else float("nan")
                print("  %8d  %14.4e  %14.4e  %12.4f"
                      % (ld, sl, nl, snr_table[ld]))

    return dict(
        field="gravity_EGM2008",
        l_nominal=2190,
        l_useful_perdegree=l_useful_pd,
        l_useful_cumulative=l_useful_ct,
        snr_below_10_deg=snr_below_10,
        snr_below_2_deg=snr_below_2,
        noise_proxy="calibrated_errors_icgem",
        snr_table=snr_table,
        max_rel_diff_crosscheck=max_rel_diff,
        S=S,
        N=N,
        notes=("EGM2008 lmax=2190 geoid spectrum; calibrated errors from ICGEM; "
               "l_min=2 (degrees 0,1 excluded from geoid); "
               "cross-check max rel diff=%.4e" % max_rel_diff),
    )


# --------------------------------------------------------------------------- #
# 3.  Magnetic (IGRF-13, SV noise proxy, lmax=13)
# --------------------------------------------------------------------------- #

def compute_magnetic_bandlimit(verbose=True):
    """Load IGRF-13 at 2020.0 and 2025.0; use the 5-year SV as a noise proxy.

    Signal S_l = per-degree power of the base field (4pi-normalised degree
    variance from the 2020.0 Gauss coefficients).

    Noise N_l = per-degree power of the 5-year change:
        N_l = sum_m ( dC_lm^2 + dS_lm^2 )   (4pi normalised)
    where dC, dS = m25.to_array() - m20.to_array().

    This is the SV-predicted 5-year drift as an uncertainty floor.

    Returns a dict with all derived quantities.
    """
    if verbose:
        print("Loading IGRF-13 (2020.0 and 2025.0)...")

    m20 = Earth.IGRF_13(lmax=13, year=2020.0)
    m25 = Earth.IGRF_13(lmax=13, year=2025.0)

    # Per-degree signal power from m20: use spectrum() which returns per-l power
    # in 4pi normalisation.  For SHMagCoeffs this returns a single array (no errors).
    sp20 = m20.spectrum()
    if isinstance(sp20, tuple):
        S = np.asarray(sp20[0], dtype=float)
    else:
        S = np.asarray(sp20, dtype=float)   # shape (14,), indices 0..13

    assert len(S) == 14, "Expected IGRF-13 spectrum length 14 (degrees 0..13)"

    # Per-degree noise from 5-year SV: difference of 4pi-normalised coefficients.
    # to_array() returns shape (2, 14, 14): arr[0] = cosine (g), arr[1] = sine (h)
    arr20 = m20.to_array()   # (2, 14, 14)
    arr25 = m25.to_array()   # (2, 14, 14)
    darr = arr25 - arr20     # coefficient differences

    # Per-degree noise: sum over orders m for each degree l
    N = np.sum(darr[0] ** 2 + darr[1] ** 2, axis=1)   # shape (14,), indices 0..13

    # l_min = 1 for B_r (no degree-0 monopole)
    l_min = 1

    # Per-degree crossover
    l_useful_pd = _perdegree_crossover(S, N, l_min=l_min)

    # Cumulative-tail crossover
    l_useful_ct = _cumulative_tail_crossover(S, N, l_min=l_min)

    l_nominal = 13

    if verbose:
        print("\n--- MAGNETIC (IGRF-13) EFFECTIVE BANDLIMIT ---")
        print("  l_nominal = 13")
        print("  Signal proxy: IGRF-13 2020.0 per-degree 4pi power")
        print("  Noise proxy : 5-year SV (2025.0 - 2020.0) per-degree 4pi power")
        print("  l_useful (per-degree SNR < 1) : %s" % l_useful_pd)
        print("  l_useful (cumulative tail)     : %s" % l_useful_ct)
        print("\n  Per-degree signal vs noise:")
        print("  %6s  %14s  %14s  %12s" % ("degree", "S_l", "N_l", "SNR_l"))
        print("  " + "-" * 52)
        for l in range(l_min, l_nominal + 1):
            snr = S[l] / N[l] if N[l] > 0 else float("inf")
            print("  %6d  %14.4e  %14.4e  %12.4f" % (l, S[l], N[l], snr))
        if l_useful_pd is None:
            print("\n  Result: field is SIGNAL-DOMINATED across the full 13-degree "
                  "band (SNR_l > 1 for all l in [1,13]); l_useful = l_nominal = 13.")
        else:
            print("\n  WARNING: crossover found at l=%d" % l_useful_pd)

    return dict(
        field="magnetic_IGRF13",
        l_nominal=l_nominal,
        l_useful_perdegree=l_useful_pd,
        l_useful_cumulative=l_useful_ct,
        snr_below_10_deg=_first_below_snr(S, N, threshold=10.0, l_min=l_min),
        snr_below_2_deg=_first_below_snr(S, N, threshold=2.0, l_min=l_min),
        noise_proxy="5yr_SV_IGRF13_2020_2025",
        S=S,
        N=N,
        notes=("IGRF-13 lmax=13; signal=4pi degree variance 2020.0; "
               "noise=5-yr SV power (2025.0-2020.0); l_min=1; blank l_useful and "
               "snr_below_10/2 mean None (signal-dominated across the band)"),
    )


# --------------------------------------------------------------------------- #
# 4.  Write results/bandlimit.csv
# --------------------------------------------------------------------------- #

def write_csv(results, outpath):
    """Write the bandlimit results dict to a CSV file.

    Parameters
    ----------
    results : list of dict
        Each dict is the return value of compute_*_bandlimit (without S, N arrays).
    outpath : str
        Absolute path to the output CSV file.
    """
    fieldnames = [
        "field", "l_useful_perdegree", "l_useful_cumulative",
        "l_nominal", "snr_below_10_deg", "snr_below_2_deg",
        "noise_proxy", "notes",
    ]
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print("\nWrote: %s" % outpath)


# --------------------------------------------------------------------------- #
# 5.  Runtime assertions (called from main)
# --------------------------------------------------------------------------- #

def _assert_gravity_invariants(res):
    """Basic sanity checks on the gravity result dict."""
    S = res["S"]
    N = res["N"]

    # Signal and noise arrays must be positive (calibrated errors => N > 0 for l>=2)
    assert np.all(S[2:] > 0), "S_l should be positive for l>=2"
    assert np.all(N[2:] > 0), "N_l (calibrated errors) should be positive for l>=2"

    # Signal at low degrees should exceed noise (well-established gravity)
    assert S[2] > N[2], ("Expected signal > noise at l=2 (largest geoid signal); "
                         "S[2]=%.4e, N[2]=%.4e" % (S[2], N[2]))

    # l_useful_perdegree must be None or an integer in [2, 2190]
    lu = res["l_useful_perdegree"]
    assert lu is None or (2 <= lu <= 2190), (
        "l_useful_perdegree out of range: %r" % lu)

    # Cross-check must be small
    assert res["max_rel_diff_crosscheck"] < 1e-6, (
        "Cross-check max rel diff too large: %.4e" % res["max_rel_diff_crosscheck"])

    # Cumulative crossover can be earlier than per-degree crossover when the
    # noise is not rapidly decaying: many future degrees each contribute noise,
    # so the total noise tail can exceed the total signal tail even while
    # individual SNR_l > 1.  We assert only that both are within [l_min, 2190].
    lc = res["l_useful_cumulative"]
    if lc is not None:
        assert 2 <= lc <= 2190, "l_useful_cumulative out of range: %r" % lc


def _assert_magnetic_invariants(res):
    """Basic sanity checks on the magnetic result dict."""
    S = res["S"]
    N = res["N"]

    # Dipole (l=1) should have the largest signal
    assert S[1] == S[1:14].max() or S[1] > S[2], (
        "Dipole (l=1) should dominate the IGRF-13 signal")

    # l_nominal = 13
    assert res["l_nominal"] == 13

    # If the field is signal-dominated across the band, l_useful_perdegree is None
    lu = res["l_useful_perdegree"]
    assert lu is None or (1 <= lu <= 13), (
        "l_useful_perdegree out of range: %r" % lu)


# --------------------------------------------------------------------------- #
# 6.  main()
# --------------------------------------------------------------------------- #

def main():
    """Run GATE C bandlimit computation and write results/bandlimit.csv."""
    print("=" * 68)
    print("GATE C — EFFECTIVE SPECTRAL BANDLIMIT (Component 1)")
    print("=" * 68)

    # --- Gravity ---
    grav = compute_gravity_bandlimit(verbose=True)
    _assert_gravity_invariants(grav)

    # --- Magnetic ---
    mag = compute_magnetic_bandlimit(verbose=True)
    _assert_magnetic_invariants(mag)

    # --- Write CSV ---
    outpath = os.path.join(C.DIR_RESULTS, "bandlimit.csv")
    write_csv([grav, mag], outpath)

    # --- Final summary ---
    print("\n" + "=" * 68)
    print("GATE C SUMMARY")
    print("=" * 68)
    print("\nGRAVITY (EGM2008, l_nominal=2190):")
    print("  l_useful per-degree  (SNR_l < 1) = %s" % grav["l_useful_perdegree"])
    print("  l_useful cumulative  (tail ratio) = %s" % grav["l_useful_cumulative"])
    print("  SNR first < 10  at degree         = %s" % grav["snr_below_10_deg"])
    print("  SNR first <  2  at degree         = %s" % grav["snr_below_2_deg"])
    print("  Cross-check max rel diff          = %.4e"
          % grav["max_rel_diff_crosscheck"])

    print("\nMAGNETIC (IGRF-13, l_nominal=13):")
    lu_mag = mag["l_useful_perdegree"]
    if lu_mag is None:
        print("  l_useful per-degree = None (signal-dominated across full band)")
    else:
        print("  l_useful per-degree = %d" % lu_mag)
    lc_mag = mag["l_useful_cumulative"]
    if lc_mag is None:
        print("  l_useful cumulative = None (signal-dominated across full band)")
    else:
        print("  l_useful cumulative = %d" % lc_mag)

    print("\nResults written to: %s" % outpath)

    # Print CSV contents for transparency
    print("\n--- bandlimit.csv contents ---")
    fieldnames = [
        "field", "l_useful_perdegree", "l_useful_cumulative",
        "l_nominal", "snr_below_10_deg", "snr_below_2_deg",
        "noise_proxy", "notes",
    ]
    rows = []
    for r in [grav, mag]:
        rows.append({k: r.get(k, "") for k in fieldnames})
    # Header
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(str(row[k]) for k in fieldnames))


if __name__ == "__main__":
    main()
