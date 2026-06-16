"""
wavelet.py -- Gate D computation: SH-vs-wavelet bit-cost comparison on a
              mid-latitude planar patch.  PLANAR FIRST CUT ONLY.

SCOPE LIMIT
-----------
This module implements the PLANAR first cut only.  Escalation to spherical /
HEALPix / spherical-wavelet methods is explicitly out of scope and is the
orchestrator's decision.  All analysis treats the extracted lat/lon patch as a
flat 2D array.

MODELLING CHOICES (auditable)
------------------------------

1.  FIELD SYNTHESIS
    The reference field is synthesised from the SHCoeffs object returned by
    shtrunc.gravity_scalar_field / magnetic_scalar_field via SHCoeffs.expand()
    on a Driscoll-Healy DH2 equiangular grid at the field's native lref
    (EGM2008 lref=360 for gravity; IGRF-13 lref=13 for magnetic).  The
    magnetic field (lmax=13, 29x57 native grid) is oversampled by padding the
    SHCoeffs object to lmax=360 with zeros before expansion; the field values
    are identical (band-limited), only the sampling density changes.  This
    brings both fields to the same 723x1445 grid with spacing ~0.249 deg/cell.

2.  PATCH EXTRACTION
    A square patch of n=128 contiguous grid cells (power of 2) is extracted
    centred at (lat=40 deg, lon=0 deg starting index).  At 0.249 deg/cell this
    is ~31.9 deg x 31.9 deg (slightly above the stated <=30 deg guideline).
    PLANAR APPROXIMATION LIMITATION: the lat/lon array is treated as a flat
    2D Euclidean array; no cosine-latitude correction is applied.  At centre
    latitude 40 deg the east-west metric distortion is cos(40 deg)=0.766, so
    the effective longitude span is ~31.9 x 0.766 = 24.4 deg.  For the purpose
    of this comparison (relative L2 error on the patch array) the planar
    treatment is consistent for both methods; neither uses the true area element.
    This approximation is explicitly stated as a limitation and does not bias
    the wavelet-vs-SH comparison.

3.  RELATIVE L2 ERROR
    eps_patch = || patch - recon ||_F / || patch ||_F  (Frobenius / plain L2
    on the flat 2D array).  This is the PATCH-RELATIVE error in both branches:
    the same denominator ensures the comparison is on equal footing.

4.  WAVELET COST  (branch A)
    Wavelet: bior4.4 (biorthogonal, 4-tap, 4 vanishing moments on analysis side;
    good compaction for piecewise smooth signals; standard choice in image
    compression literature for natural images).  Boundary mode: 'periodization'
    (enforces total coefficient count N = n^2 = 16384, eliminating boundary
    expansion artefacts and making the index overhead formula exact).  Level:
    full decomposition (pywt default for power-of-2 size gives 7 levels for
    n=128 with periodization).

    Thresholding: keep the K largest-magnitude coefficients (hard threshold).
    K is found by binary bisection on the exact reconstruction error until
    eps_patch <= eps_target, searching over K in [1, N].

    Bit cost:
        bits_wavelet = K * b_q  +  K * ceil(log2(N))
    where:
      - K * b_q  is the payload (K coefficient values at b_q bits each)
      - K * ceil(log2(N))  is the SPARSE INDEX OVERHEAD: storing WHICH K of
        the N positions are nonzero requires ceil(log2(N)) bits per kept
        coefficient (one position index per coefficient, no run-length).  This
        is a simple unordered index list; entropy-coded alternatives exist but
        are not modelled here.  The overhead is INCLUDED because wavelets are
        sparse (K << N at small eps) and SH is DENSE (no index cost up to L);
        omitting the overhead would overstate the wavelet advantage.
    N = n^2 = 16384,  ceil(log2(16384)) = 14 bits per index.

5.  SH COST  (branch B)
    Find the smallest INTEGER degree L such that the SH-truncated field's
    patch-relative L2 error satisfies eps_patch <= eps_target.  Procedure:
      - Synthesise the reference patch from the full-L SHCoeffs.
      - For a dense set of candidate L values (logspaced for the geoid; all
        integers 1..13 for the magnetic field), synthesise the truncated field
        on the patch and measure eps_patch.
      - Interpolate (log-log) to find the crossing L_cross; set L = ceil(L_cross).
      - Verify the exact L with a final synthesis.

    Global SH storage: (L+1)^2 coefficients at b_q bits each (dense, no index
    overhead needed up to degree L).
        bits_SH_global = (L+1)^2 * b_q

    AMORTISED patch cost: the SH expansion is global; to compare fairly against
    the patch-local wavelet cost, the global coefficient set is amortised by the
    fraction of solid angle covered by the patch:
        f = patch_solid_angle / (4 * pi)
        patch_solid_angle = (sin(lat_hi_rad) - sin(lat_lo_rad)) * patch_lon_rad
    where lat_hi, lat_lo are the patch latitude bounds and patch_lon is the
    longitude span in radians.
        bits_SH_patch = (L+1)^2 * b_q * f

    STATEMENT OF MODELLING CHOICE (key for the orchestrator to audit):
    The amortisation is the central modelling choice.  A decoder that only needs
    to reconstruct a small patch of the field might argue that it does not need
    all (L+1)^2 coefficients -- only those whose support overlaps the patch.
    In practice, SH basis functions are globally supported so ALL (L+1)^2
    coefficients are needed (no locality), making bits_SH_global the honest
    cost for any patch reconstruction from a global SH model.  The amortised
    figure bits_SH_patch is the *effective per-patch cost* under the assumption
    that the storage cost is shared uniformly across the sphere (i.e. the same
    global model is used to reconstruct many equal-area patches).  BOTH figures
    are reported; the orchestrator must decide which is the right comparand.
    For the Gate D conclusion, the amortised figure is used (as specified in the
    task), clearly labelled.

6.  QUANTIZER BIT-DEPTH
    b_q = 16 bits for both branches.  The quantization contribution to eps is
    (dynamic_range / (2^b_q * sqrt(12))) / rms_patch.  For the geoid patch
    (range ~100 m, rms ~30 m) this is ~1.5e-5; for the magnetic patch
    (~120000 nT range, ~20000 nT rms) ~2.6e-5.  Both are <<< eps_target=0.01,
    so 16-bit quantization does not dominate the comparison at either eps value.
    This is verified and reported per field/eps in the output.

7.  SOLID-ANGLE FRACTION
    Computed exactly from the grid-derived patch latitude bounds (using actual
    sampled lat values from the DH2 grid), not from an approximate formula.

8.  FIELDS ANALYSED
    - gravity  : EGM2008 geoid undulation N (m), lref=360, SHCoeffs from
                 shtrunc.gravity_scalar_field().  DH2 grid at lref=360.
    - magnetic : IGRF-13 B_r (nT), lref=13, SHCoeffs from
                 shtrunc.magnetic_scalar_field(), oversampled to DH2 lref=360
                 grid via zero-padding in SH space.  Patch-eps scan over L=1..13.

References
----------
Cohen, A., Daubechies, I., & Feauveau, J.-C. (1992). Biorthogonal bases of
    compactly supported wavelets. Communications on Pure and Applied
    Mathematics, 45(5), 485-560.
Mallat, S. (1999). A wavelet tour of signal processing. Academic Press.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pywt

# Import read-only from project modules; do NOT modify shtrunc or config.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config as C
import shtrunc

# ---------------------------------------------------------------------------
# Constants (modelling choices — all documented in module docstring)
# ---------------------------------------------------------------------------
WAVELET = "bior4.4"           # wavelet family
BOUNDARY_MODE = "periodization"  # enforces N_coeffs = n^2 exactly
B_Q = 16                      # quantiser bit-depth (same for both branches)
PATCH_N = 128                 # patch side length (power of 2)
PATCH_LAT_CENTRE = 40.0       # degrees N
EPS_TARGETS = [0.05, 0.01]    # relative L2 error thresholds
FIELDS = ["gravity", "magnetic"]

# ---------------------------------------------------------------------------
# 1.  Synthesise field on DH2 grid and extract mid-latitude patch
# ---------------------------------------------------------------------------

def _dh2_data_and_lats_lons(s_coeffs, lref_grid):
    """Expand SHCoeffs on a DH2 grid at lref_grid; return (data, lats, lons).

    For oversampling (e.g. lmax=13 coeffs on a lmax=360 grid), the coeffs are
    padded to lref_grid with zeros first.  The result is the same field at
    higher spatial sampling density.
    """
    if s_coeffs.lmax != lref_grid:
        s_padded = s_coeffs.pad(lref_grid)
    else:
        s_padded = s_coeffs
    grid_obj = s_padded.expand(grid="DH2")
    return grid_obj.data, grid_obj.lats(), grid_obj.lons()


def extract_patch(data, lats, lons, lat_centre, n):
    """Extract an n x n patch centred (approx) at lat_centre from data.

    Returns (patch, lat_lo, lat_hi, lon_lo, lon_hi, row_start, col_start).
    The patch is taken from consecutive grid rows/columns, so it is a contiguous
    lat/lon block.  The column start is chosen so that the patch is roughly
    centred in longitude (avoids map edges at lon=0/360 wrap).
    """
    nlat, nlon = data.shape
    # Find row index closest to lat_centre
    row_centre = int(np.argmin(np.abs(lats - lat_centre)))
    row_start = row_centre - n // 2
    row_end = row_start + n
    # Guard bounds
    row_start = max(0, min(row_start, nlat - n))
    row_end = row_start + n

    # Centre the column around lon=180 to avoid the 0/360 meridian boundary.
    col_centre = int(np.argmin(np.abs(lons - 180.0)))
    col_start = col_centre - n // 2
    col_end = col_start + n
    col_start = max(0, min(col_start, nlon - n))
    col_end = col_start + n

    patch = data[row_start:row_end, col_start:col_end].copy()

    lat_lo = float(lats[row_end - 1])    # lats are descending (90...-90)
    lat_hi = float(lats[row_start])
    lon_lo = float(lons[col_start])
    lon_hi = float(lons[col_end - 1])

    return patch, lat_lo, lat_hi, lon_lo, lon_hi, row_start, col_start


def patch_solid_angle_fraction(lat_lo_deg, lat_hi_deg, lon_lo_deg, lon_hi_deg):
    """Solid angle of a lat/lon patch as a fraction of 4*pi steradians.

    solid_angle = (sin(lat_hi) - sin(lat_lo)) * (lon_hi - lon_lo)  [radians]
    fraction = solid_angle / (4*pi)
    """
    lat_lo_r = math.radians(lat_lo_deg)
    lat_hi_r = math.radians(lat_hi_deg)
    dlon_r = math.radians(abs(lon_hi_deg - lon_lo_deg))
    omega = (math.sin(lat_hi_r) - math.sin(lat_lo_r)) * dlon_r
    return float(omega / (4.0 * math.pi))


# ---------------------------------------------------------------------------
# 2.  Relative L2 error (patch-local, flat 2D Frobenius)
# ---------------------------------------------------------------------------

def rel_l2_error(ref, recon):
    """||ref - recon||_F / ||ref||_F on flat 2D arrays."""
    denom = np.sqrt(np.sum(ref ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sqrt(np.sum((ref - recon) ** 2)) / denom)


# ---------------------------------------------------------------------------
# 3.  Wavelet cost (branch A)
# ---------------------------------------------------------------------------

def _wavelet_K_for_eps(patch, eps_target, wavelet=WAVELET, mode=BOUNDARY_MODE):
    """Find minimum K (number of kept coefficients) such that reconstruction
    achieves relative L2 error <= eps_target on the patch.

    Uses binary bisection over K for exact (not Parseval-approximate) error.
    Returns (K, final_eps, total_coeffs, coeffs_flat, slices).
    """
    coeffs = pywt.wavedec2(patch, wavelet, level=None, mode=mode)
    arr, slices = pywt.coeffs_to_array(coeffs)
    flat = arr.ravel().copy()
    N = flat.size
    idx_sorted = np.argsort(np.abs(flat))[::-1]  # descending magnitude

    def _recon_eps(K):
        mask = np.zeros(N, dtype=bool)
        mask[idx_sorted[:K]] = True
        arr_thresh = (flat * mask).reshape(arr.shape)
        c_thresh = pywt.array_to_coeffs(arr_thresh, slices, output_format="wavedec2")
        rec = pywt.waverec2(c_thresh, wavelet, mode=mode)
        # For periodization mode, shape matches exactly
        rec = rec[: patch.shape[0], : patch.shape[1]]
        return rel_l2_error(patch, rec)

    # Binary bisection: find smallest K such that _recon_eps(K) <= eps_target
    lo, hi = 1, N
    # Check if full reconstruction is already within eps (sanity guard)
    # hi=N gives perfect reconstruction (eps~0); lo=1 likely has large error.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _recon_eps(mid) <= eps_target:
            hi = mid
        else:
            lo = mid
    K = hi
    final_eps = _recon_eps(K)
    return K, final_eps, N, flat, slices


def wavelet_bits(K, N, b_q=B_Q):
    """Bit cost for sparse wavelet representation.

    bits = K * b_q  +  K * ceil(log2(N))
    First term: payload (K coefficient values at b_q bits each).
    Second term: sparse index overhead (store WHICH K of N positions are kept).
    """
    index_bits_per_coeff = math.ceil(math.log2(N)) if N > 1 else 1
    return int(K) * b_q + int(K) * index_bits_per_coeff


# ---------------------------------------------------------------------------
# 4.  SH cost (branch B): scan truncation degrees on the patch
# ---------------------------------------------------------------------------

def _sh_truncated_patch(s_coeffs, L, lref_grid, row_start, col_start, n):
    """Synthesise the degree-L truncated field on the patch.

    Returns the n x n patch array for the truncated SHCoeffs.
    """
    s_trunc = shtrunc.truncate(s_coeffs, L)
    data, lats, lons = _dh2_data_and_lats_lons(s_trunc, lref_grid)
    return data[row_start: row_start + n, col_start: col_start + n].copy()


def find_sh_L_for_eps(s_coeffs, ref_patch, lref_grid,
                      row_start, col_start, n, eps_target):
    """Find smallest integer L such that SH truncation to L achieves
    patch-relative L2 error <= eps_target.

    Strategy:
    - For magnetic (lref<=13): scan L=1..lref exhaustively.
    - For geoid (lref=360): scan logspaced L values, log-log interpolate
      the crossing, then verify the ceiling integer.

    Returns (L, final_eps, eps_scan_L, eps_scan_eps).
    """
    lref = s_coeffs.lmax

    if lref <= 20:
        # Dense scan (magnetic: lref=13)
        L_scan = list(range(1, lref + 1))
    else:
        # Logspaced scan (gravity: lref=360) — dense enough to interpolate well
        pts = np.unique(np.round(np.geomspace(2, lref, 80)).astype(int))
        L_scan = sorted(set(list(pts) + [lref]))

    eps_scan = []
    for L_val in L_scan:
        patch_trunc = _sh_truncated_patch(s_coeffs, L_val, lref_grid,
                                          row_start, col_start, n)
        eps_scan.append(rel_l2_error(ref_patch, patch_trunc))

    L_scan = np.array(L_scan, dtype=int)
    eps_scan = np.array(eps_scan, dtype=float)

    # Find crossing: smallest L where eps_scan <= eps_target
    crossing_idx = np.where(eps_scan <= eps_target)[0]
    if len(crossing_idx) == 0:
        # eps never reaches target within lref
        L_found = int(lref)
        final_eps = float(eps_scan[-1])
    else:
        first_idx = int(crossing_idx[0])
        if first_idx == 0:
            L_found = int(L_scan[0])
            final_eps = float(eps_scan[0])
        else:
            # Log-log interpolate between L_scan[first_idx-1] and L_scan[first_idx]
            e0, e1 = eps_scan[first_idx - 1], eps_scan[first_idx]
            l0, l1 = float(L_scan[first_idx - 1]), float(L_scan[first_idx])
            if e0 > 0 and e1 > 0 and l0 > 0 and l1 > 0:
                frac = ((math.log10(e0) - math.log10(eps_target))
                        / (math.log10(e0) - math.log10(e1)))
                l_cross = 10 ** (math.log10(l0) + frac * (math.log10(l1) - math.log10(l0)))
            else:
                l_cross = l0 + (l1 - l0) * (e0 - eps_target) / (e0 - e1)
            L_found = int(math.ceil(l_cross))
            # Clamp to valid range
            L_found = max(1, min(L_found, int(lref)))
            # Verify with exact synthesis
            patch_verify = _sh_truncated_patch(s_coeffs, L_found, lref_grid,
                                               row_start, col_start, n)
            final_eps = rel_l2_error(ref_patch, patch_verify)
            # If the interpolated L overshoots (can happen at coarse scan), step back
            if final_eps > eps_target and L_found < lref:
                # Binary search between L_found and L_scan[first_idx]
                lo_L, hi_L = L_found, int(L_scan[first_idx])
                while hi_L - lo_L > 1:
                    mid_L = (lo_L + hi_L) // 2
                    p = _sh_truncated_patch(s_coeffs, mid_L, lref_grid,
                                            row_start, col_start, n)
                    if rel_l2_error(ref_patch, p) <= eps_target:
                        hi_L = mid_L
                    else:
                        lo_L = mid_L
                L_found = hi_L
                patch_verify = _sh_truncated_patch(s_coeffs, L_found, lref_grid,
                                                   row_start, col_start, n)
                final_eps = rel_l2_error(ref_patch, patch_verify)

    return int(L_found), float(final_eps), L_scan, eps_scan


def sh_bits(L, b_q=B_Q):
    """Global SH storage: (L+1)^2 coefficients at b_q bits."""
    return int((L + 1) ** 2) * b_q


def sh_bits_patch(L, frac, b_q=B_Q):
    """Amortised patch SH cost: bits_SH_global * solid_angle_fraction."""
    return sh_bits(L, b_q) * frac


# ---------------------------------------------------------------------------
# 5.  Quantisation error check
# ---------------------------------------------------------------------------

def quantisation_eps(patch, b_q=B_Q):
    """Relative L2 quantisation error for a uniform b_q-bit scalar quantiser.

    eps_quant = (dynamic_range / (2^b_q * sqrt(12))) / rms_patch
    where dynamic_range = max(patch) - min(patch) and rms_patch = ||patch||_F / n.
    This is the worst-case (white-noise) quantisation error relative to the
    patch norm; actual entropy-coded quantisation would be smaller.
    """
    dr = float(patch.max() - patch.min())
    rms = float(np.sqrt(np.mean(patch ** 2)))
    if rms == 0.0:
        return 0.0
    step = dr / (2 ** b_q)
    noise_rms = step / math.sqrt(12.0)
    return float(noise_rms / rms)


# ---------------------------------------------------------------------------
# 6.  Per-field runner
# ---------------------------------------------------------------------------

def _load_field(field_name):
    """Load the SHCoeffs and metadata for the named field."""
    if field_name == "gravity":
        fd = shtrunc.gravity_scalar_field()
    elif field_name == "magnetic":
        fd = shtrunc.magnetic_scalar_field()
    else:
        raise ValueError("field_name must be 'gravity' or 'magnetic', got %r"
                         % field_name)
    return fd


def run_field(field_name, eps_targets=None, verbose=True):
    """Run the full wavelet-vs-SH comparison for one field.

    Returns a dict: field_name -> {eps: result_dict}.
    """
    if eps_targets is None:
        eps_targets = EPS_TARGETS

    if verbose:
        print("\n" + "=" * 60)
        print("Field: %s" % field_name)
        print("=" * 60)

    fd = _load_field(field_name)
    s = fd["s"]

    # Grid resolution: always synthesise at GRAV_LREF=360 for both fields
    lref_grid = C.GRAV_LREF  # 360

    if verbose:
        print("  Loading DH2 grid (lref_grid=%d)..." % lref_grid)

    data, lats, lons = _dh2_data_and_lats_lons(s, lref_grid)
    patch, lat_lo, lat_hi, lon_lo, lon_hi, row_start, col_start = \
        extract_patch(data, lats, lons, PATCH_LAT_CENTRE, PATCH_N)

    # Grid cell spacing (degrees)
    dlat = abs(float(lats[0]) - float(lats[1]))
    patch_deg = PATCH_N * dlat  # approximate patch width in degrees

    # Solid-angle fraction
    frac = patch_solid_angle_fraction(lat_lo, lat_hi, lon_lo, lon_hi)

    if verbose:
        print("  Patch: %dx%d cells, centre lat=%.1f deg" % (PATCH_N, PATCH_N, PATCH_LAT_CENTRE))
        print("  Patch bounds: lat=[%.2f, %.2f], lon=[%.2f, %.2f] deg"
              % (lat_lo, lat_hi, lon_lo, lon_hi))
        print("  Patch width: ~%.1f deg, solid-angle fraction: %.5f" % (patch_deg, frac))
        print("  b_q = %d bits" % B_Q)

    per_eps = {}
    for eps_target in eps_targets:
        if verbose:
            print("\n  -- eps_target = %.2f --" % eps_target)

        # --- Quantisation error check ---
        q_err = quantisation_eps(patch, B_Q)

        # --- Wavelet branch (A) ---
        if verbose:
            print("  [A] Wavelet (%s, %s): bisecting K..." % (WAVELET, BOUNDARY_MODE))
        K, wav_final_eps, N_coeffs, _, _ = _wavelet_K_for_eps(patch, eps_target)
        bits_wav = wavelet_bits(K, N_coeffs, B_Q)
        idx_bits_per = math.ceil(math.log2(N_coeffs)) if N_coeffs > 1 else 1

        if verbose:
            print("    K=%d / %d  eps=%.6f  bits_wav=%d" % (K, N_coeffs, wav_final_eps, bits_wav))
            print("    (payload: %d bits, index overhead: %d bits)"
                  % (K * B_Q, K * idx_bits_per))

        # --- SH branch (B) ---
        if verbose:
            print("  [B] SH truncation: scanning degrees...")
        L, sh_final_eps, L_scan, eps_sh_scan = find_sh_L_for_eps(
            s, patch, lref_grid, row_start, col_start, PATCH_N, eps_target)
        bits_sh_global = sh_bits(L, B_Q)
        bits_sh_patch = sh_bits_patch(L, frac, B_Q)

        if verbose:
            print("    L=%d  eps=%.6f  bits_SH_global=%d  bits_SH_patch=%.1f"
                  % (L, sh_final_eps, bits_sh_global, bits_sh_patch))

        # --- Ratio ---
        ratio = bits_wav / bits_sh_patch if bits_sh_patch > 0 else float("inf")

        if verbose:
            print("    bits_wavelet / bits_SH_patch = %.3f  (%s wins)"
                  % (ratio, "wavelet" if ratio < 1 else "SH"))
            print("    quant_err_check: %.2e  (<<eps? %s)"
                  % (q_err, "YES" if q_err < eps_target / 10 else "MARGINAL"))

        per_eps[str(eps_target)] = dict(
            eps_target=eps_target,
            # Wavelet
            K_wavelet=int(K),
            N_total_coeffs=int(N_coeffs),
            wavelet_final_eps=float(wav_final_eps),
            bits_wavelet=int(bits_wav),
            payload_bits=int(K * B_Q),
            index_overhead_bits=int(K * idx_bits_per),
            index_bits_per_coeff=int(idx_bits_per),
            # SH
            L_sh=int(L),
            sh_final_eps=float(sh_final_eps),
            bits_SH_global=int(bits_sh_global),
            bits_SH_patch=float(bits_sh_patch),
            # Comparison
            wavelet_over_sh_patch_ratio=float(ratio),
            winner="wavelet" if ratio < 1 else "SH",
            # Meta
            b_q=int(B_Q),
            patch_n=int(PATCH_N),
            patch_deg=float(patch_deg),
            patch_lat_centre=float(PATCH_LAT_CENTRE),
            lat_lo=float(lat_lo),
            lat_hi=float(lat_hi),
            lon_lo=float(lon_lo),
            lon_hi=float(lon_hi),
            solid_angle_fraction=float(frac),
            wavelet=WAVELET,
            boundary_mode=BOUNDARY_MODE,
            quant_err_check=float(q_err),
            quant_dominates=bool(q_err >= eps_target / 10),
        )

    return per_eps


# ---------------------------------------------------------------------------
# 7.  Print summary table
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a human-readable comparison table."""
    hdr = ("%-10s  %-6s  %8s  %12s  %6s  %12s  %12s  %8s  %-8s  %-9s"
           % ("Field", "eps", "K_wave", "bits_wave", "L_SH",
              "bits_SH_patch", "bits_SH_glob", "ratio", "winner", "q_err"))
    print("\n" + "-" * len(hdr))
    print("Gate D — Planar Wavelet-vs-SH Bit-Cost Comparison")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for field_name in FIELDS:
        if field_name not in results:
            continue
        for eps_str, r in sorted(results[field_name].items(),
                                 key=lambda kv: float(kv[0]), reverse=True):
            print(
                "%-10s  %-6s  %8d  %12d  %6d  %12.0f  %12d  %8.3f  %-8s  %9.2e"
                % (field_name, eps_str,
                   r["K_wavelet"], r["bits_wavelet"],
                   r["L_sh"], r["bits_SH_patch"], r["bits_SH_global"],
                   r["wavelet_over_sh_patch_ratio"], r["winner"],
                   r["quant_err_check"])
            )
    print("-" * len(hdr))
    print("Columns: K_wave=kept coefficients; ratio=bits_wavelet/bits_SH_patch")
    print("bits_SH_patch = bits_SH_global * solid_angle_fraction (amortised)")
    print("bits_wavelet  = K*b_q + K*ceil(log2(N_total_coeffs)) (payload + index)")
    print("b_q = %d bits;  wavelet = %s (%s)" % (B_Q, WAVELET, BOUNDARY_MODE))


# ---------------------------------------------------------------------------
# 8.  Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run the Gate D comparison and write results/wavelet_comparison.json."""
    results = {}
    for field_name in FIELDS:
        results[field_name] = run_field(field_name, EPS_TARGETS, verbose=True)

    print_table(results)

    # Save JSON
    out_path = os.path.join(C.DIR_RESULTS, "wavelet_comparison.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nResults written to: %s" % out_path)
    return results


if __name__ == "__main__":
    main()
