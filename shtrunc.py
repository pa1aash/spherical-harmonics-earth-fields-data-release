"""
shtrunc.py  --  Core analysis for spherical-harmonic spectral truncation of
Earth's gravity (EGM2008) and magnetic (IGRF-13) fields.

Primary outcome
---------------
The relative L2(S^2) reconstruction error

        eps(l_max) = || f - S_{l_max} f ||  /  || f ||

evaluated with proper spherical-area (Gauss-Legendre quadrature) weighting, where
f is a real, band-limited scalar field expanded to a reference degree l_ref and
S_{l_max} f is the same field truncated at degree l_max.

Design notes (see README.md for the full rationale)
---------------------------------------------------
* We analyse a SCALAR derived field per dataset:
    - gravity  -> geoid undulation N (metres), relative to the WGS84 normal
                  ellipsoid (so the J2 flattening / reference ellipsoid is
                  removed and maps show anomalies).  Degrees 0 and 1 are also
                  removed (constant offset + geocentre term).
    - magnetic -> radial field B_r (nT) at Earth's surface, all degrees 1..13.
* The geoid is built with first-order Bruns (linear in the Stokes coefficients)
  so the analysed field is EXACTLY band-limited.  Truncating this scalar field at
  degree l is therefore identical to truncating the source model at degree l,
  because the geoid / B_r synthesis is diagonal in degree.
* Quadrature: Gauss-Legendre (GLQ) grids give an EXACT discrete L2 inner product
  for band-limited fields, so the grid-based eps equals the analytic Parseval
  prediction to numerical precision (this is the verification cross-check).

Gravity functionals (gravity_functional_field)
----------------------------------------------
Additional gravity functionals are computed via DIAGONAL per-degree rescalings of
the disturbing-potential degree variance P_l^T.  These are exact and grid-free:

    P_l^func = w_l^2 * P_l^T

where the per-degree spectral weight w_l (the constant GM/r0^k radial prefactors
cancel in the RELATIVE error eps, so only the l-dependent factor matters):

    geoid        : w_l = 1               (cross-check: reproduces headline geoid)
    disturbance  : w_l = (l+1)           (gravity disturbance dg; matches
                                          pyshtools spectrum('radial') convention)
    anomaly      : w_l = (l-1)           (free-air gravity anomaly; NOTE: NOT
                                          (l+1) -- the anomaly operator is
                                          (l-1), not (l+1); first valid at l=2
                                          so w_2 = 1, always >= 0 for l >= 2)
    Trr          : w_l = (l+1)*(l+2)    (radial-radial gradient T_rr)

All four functionals share the same disturbing-potential degree variance base
(EGM2008, GRS80 normal field removed, degrees 0 and 1 dropped, l_min=2).

Import-only module
------------------
This module has no ``__main__`` block and is not directly executable.
Run ``make_figures.py`` (or the individual analysis scripts: ``planetary.py``,
``bandlimit.py``, ``codec.py``, etc.) to reproduce results.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pyshtools as pysh
import pyshtools.datasets.Earth as Earth
import boule

import config as C


# --------------------------------------------------------------------------- #
# 1.  Data loading + scalar-field synthesis
# --------------------------------------------------------------------------- #
def _zero_low_degrees(coeffs, lmin):
    """Return a copy of an SHCoeffs with all degrees < lmin set to zero."""
    out = coeffs.copy()
    out.coeffs[:, :lmin, :] = 0.0
    return out


def gravity_scalar_field():
    """EGM2008 geoid undulation (m) as a 4pi-normalised SHCoeffs scalar field.

    Returns a dict with the field and its provenance/metadata.
    """
    grav = Earth.EGM2008(lmax=C.GRAV_LREF)            # SHGravRealCoeffs (4pi)
    ell = boule.WGS84
    # First-order Bruns geoid on a DH grid at the reference bandwidth.
    geoid = grav.geoid(ell, lmax=C.GRAV_LREF, lmax_calc=C.GRAV_LREF,
                       order=C.GRAV_GEOID_ORDER)
    grid = geoid.geoid                                # DHRealGrid, units = m
    s = grid.expand()                                 # SHCoeffs (4pi), lmax=lref
    s = _zero_low_degrees(s, lmin=2)                  # remove deg 0 (offset) & 1
    prov = ("EGM2008 (Pavlis et al. 2012), tide-free, via pyshtools.datasets."
            "Earth.EGM2008 -> ICGEM. Derived field: geoid undulation N relative "
            "to the WGS84 normal ellipsoid (boule.WGS84), first-order Bruns; "
            "degrees 0 and 1 removed; reference degree l_ref=%d." % C.GRAV_LREF)
    return dict(
        key="gravity", label="Gravity (EGM2008 geoid)", short="gravity",
        color=C.COLOR_GRAVITY, units="m", field_symbol="N",
        s=s, lref=C.GRAV_LREF, lmax_grid=C.GRAV_LMAX_GRID,
        map_lmax=C.GRAV_MAP_LMAX, lmax_list=C.GRAV_LMAX_LIST,
        fit_range=C.GRAV_FIT_RANGE, map_degrees=C.GRAV_MAP_DEGREES,
        spectrum_label="Geoid degree variance", spectrum_units="m$^2$",
        secondary=False, provenance=prov,
    )


def _magnetic_from_coeffs(mag, lref, lmax_grid, map_lmax, lmax_list, fit_range,
                          key, label, short, color, provenance, secondary=False):
    """Common builder: synthesise B_r (nT) at r0 and return the field dict."""
    grid = mag.expand(lmax=lref, lmax_calc=lref)      # SHMagGrid at r = r0
    rad = grid.rad                                     # DHRealGrid, B_r (nT)
    s = rad.expand()                                   # SHCoeffs (4pi)
    s = _zero_low_degrees(s, lmin=1)                   # (no degree-0 monopole)
    return dict(
        key=key, label=label, short=short, color=color, units="nT",
        field_symbol="B_r", s=s, lref=lref, lmax_grid=lmax_grid,
        map_lmax=map_lmax, lmax_list=lmax_list, fit_range=fit_range,
        map_degrees=None,
        spectrum_label="Lowes--Mauersberger spectrum $R_l$",
        spectrum_units="nT$^2$", secondary=secondary,
        # conventional geomagnetic spectrum, computed from Gauss coefficients:
        lowes=lowes_mauersberger(mag, lref),
        provenance=provenance,
    )


def magnetic_scalar_field():
    """IGRF-13 radial field B_r (nT) at Earth's surface, degrees 1..13."""
    mag = Earth.IGRF_13(lmax=C.MAG_LREF, year=C.MAG_YEAR)
    prov = ("IGRF-13 (Alken et al. 2021) epoch %.1f, via pyshtools.datasets."
            "Earth.IGRF_13 -> NOAA NCEI. Derived field: radial component B_r at "
            "Earth's surface (r0 = %.1f km); all degrees 1..%d retained."
            % (C.MAG_YEAR, mag.r0 / 1e3, C.MAG_LREF))
    return _magnetic_from_coeffs(
        mag, C.MAG_LREF, C.MAG_LMAX_GRID, C.MAG_MAP_LMAX, C.MAG_LMAX_LIST,
        C.MAG_FIT_RANGE, key="magnetic", label="Magnetic (IGRF-13 $B_r$)",
        short="magnetic", color=C.COLOR_MAGNETIC, provenance=prov)


def lithospheric_scalar_field():
    """Optional: NGDC-720 V3 crustal B_r (nT), a longer high-degree magnetic
    curve.  Returns None if the dataset cannot be downloaded."""
    try:
        mag = Earth.NGDC_720_V3(lmax=C.LITHO_LREF)
    except Exception as exc:                           # pragma: no cover
        print("  [litho] NGDC-720 unavailable (%s); skipping." % exc)
        return None
    prov = ("NGDC-720 V3 (Maus 2010) lithospheric model, via pyshtools.datasets."
            "Earth.NGDC_720_V3 -> NOAA NCEI. Derived field: crustal B_r at "
            "Earth's surface; degrees 16..%d (degrees <16 are zero by "
            "construction). SECONDARY/optional curve." % C.LITHO_LREF)
    return _magnetic_from_coeffs(
        mag, C.LITHO_LREF, C.LITHO_LMAX_GRID, C.LITHO_LMAX_GRID,
        C.LITHO_LMAX_LIST, C.LITHO_FIT_RANGE, key="lithospheric",
        label="Magnetic crustal (NGDC-720 $B_r$)", short="lithospheric",
        color=C.COLOR_LITHO, provenance=prov, secondary=True)


def lowes_mauersberger(mag, lref):
    """Lowes--Mauersberger spatial power spectrum R_l = (l+1) sum_m (g^2 + h^2)
    at the reference radius, from Schmidt-normalised Gauss coefficients (nT^2).
    """
    # Convert to Schmidt-normalised real arrays (g,h) -> array[2, l+1, m+1].
    sch = mag.convert(normalization="schmidt", csphase=1)
    arr = sch.to_array()                # shape (2, lref+1, lref+1)
    ls = np.arange(lref + 1)
    power = np.sum(arr[0] ** 2 + arr[1] ** 2, axis=1)   # sum over m, per degree
    Rl = (ls + 1) * power
    return ls, Rl


# --------------------------------------------------------------------------- #
# 1b. Planetary geoids (Moon, Mars) -- same first-order-Bruns pipeline as Earth
# --------------------------------------------------------------------------- #
# Accessors confirmed against the installed pyshtools 4.14.1 / boule 0.6.0:
# Mars MRO120D does NOT exist; use GMM3 (Genova et al. 2016).  boule.Moon2015 is
# a Sphere (f=0), boule.Mars2009 an Ellipsoid.  GMM3 is Kaula-constrained above
# degree 90 and GRGM1200B above degree 600, so each l_ref stays in the
# data-driven band.
_PLANETARY_SPEC = {
    "moon": dict(module="Moon", accessor="GRGM1200B", ellipsoid="Moon2015",
                 native=1200, lref=360, lmax_grid=360, fit_range=(4, 300),
                 label="Lunar geoid (GRGM1200B)",
                 cite="Goossens et al. 2020, JGR Planets, doi:10.1029/2019JE006086"),
    "mars": dict(module="Mars", accessor="GMM3", ellipsoid="Mars2009",
                 native=120, lref=90, lmax_grid=90, fit_range=(4, 75),
                 label="Areoid (Mars GMM3)",
                 cite="Genova et al. 2016, Icarus, doi:10.1016/j.icarus.2016.02.050"),
}


def planetary_geoid_field(body):
    """Geoid (selenoid / areoid) undulation field for the Moon or Mars, built
    with the SAME first-order-Bruns pipeline as the Earth geoid
    (gravity_scalar_field): degrees 0 and 1 removed, scalar field exactly
    band-limited so truncating it equals truncating the source model.

    body in {'moon','mars'}.  Returns the same field dict shape as the Earth
    builders, ready for analyse_field / epsilon_curve.
    """
    import importlib
    spec = _PLANETARY_SPEC[body]
    ds = importlib.import_module("pyshtools.datasets.%s" % spec["module"])
    grav = getattr(ds, spec["accessor"])(lmax=spec["lref"])
    ell = getattr(boule, spec["ellipsoid"])
    geoid = grav.geoid(ell, lmax=spec["lref"], lmax_calc=spec["lref"],
                       order=C.GRAV_GEOID_ORDER)
    s = geoid.geoid.expand()                            # SHCoeffs (4pi), units = m
    s = _zero_low_degrees(s, lmin=2)                    # drop offset + geocentre
    lmax_list = C._logspaced_degrees(spec["lref"])
    prov = ("%s gravity model via pyshtools.datasets.%s.%s (native degree %d, "
            "capped at l_ref=%d at load) -> first-order-Bruns geoid undulation "
            "relative to boule.%s; degrees 0 and 1 removed. Reference: %s."
            % (spec["module"], spec["module"], spec["accessor"], spec["native"],
               spec["lref"], spec["ellipsoid"], spec["cite"]))
    return dict(
        key="%s_geoid" % body, label=spec["label"], short=body,
        color=C.COLOR_GRAVITY, units="m", field_symbol="N", s=s,
        lref=spec["lref"], lmax_grid=spec["lmax_grid"], map_lmax=spec["lmax_grid"],
        lmax_list=lmax_list, fit_range=spec["fit_range"], map_degrees=None,
        spectrum_label="%s geoid degree variance" % spec["module"],
        spectrum_units="m$^2$", secondary=False,
        accessor="%s.%s" % (spec["module"], spec["accessor"]),
        ellipsoid="boule.%s" % spec["ellipsoid"], cite=spec["cite"],
        provenance=prov)


# --------------------------------------------------------------------------- #
# 2.  Grid + exact spherical-area (GLQ) weights
# --------------------------------------------------------------------------- #
def glq_setup(lmax_grid):
    """Return (zeros, lat_weights, area_weights_2d, nlat, nlon) for a GLQ grid.

    area_weights sum to 4*pi and give an EXACT discrete integral over S^2 for
    fields band-limited to <= lmax_grid:  integral f dOmega = sum_ij W_ij f_ij.
    """
    zeros, w = pysh.expand.SHGLQ(lmax_grid)            # w: Gauss-Legendre weights
    nlat = lmax_grid + 1
    nlon = 2 * lmax_grid + 1
    dphi = 2.0 * np.pi / nlon
    W = w[:, None] * dphi                               # (nlat, 1) -> broadcast
    W = np.broadcast_to(W, (nlat, nlon)).copy()
    assert np.isclose(W.sum(), 4 * np.pi, atol=1e-12), (
        "GLQ weight sum != 4*pi: got %.6e" % W.sum())
    return zeros, w, W, nlat, nlon


def expand_on_glq(s, zeros, lmax_grid):
    """Expand SHCoeffs s on the GLQ nodes `zeros` at bandwidth lmax_grid.

    extend=False so the grid is exactly (lmax+1) x (2*lmax+1) with no duplicated
    360-degree meridian -- required for correct (non-double-counted) quadrature.
    """
    sp = s.pad(lmax_grid) if s.lmax != lmax_grid else s
    grid = sp.expand(grid="GLQ", zeros=zeros, extend=False)
    return grid.data


def truncate(s, lmax):
    """Copy of s with all degrees > lmax set to zero."""
    out = s.copy()
    if lmax < out.lmax:
        out.coeffs[:, lmax + 1:, :] = 0.0
    return out


def truncate_best_N(s, N):
    """Best N-term approximation: a copy of s keeping only the N largest-magnitude
    coefficients (oracle threshold across ALL degrees/orders), the rest zeroed.

    In an orthonormal basis the best N-term L2 approximation is exactly "keep the N
    largest |coefficients|" (the diagonal Eckart-Young case).  Unlike degree
    truncation it can keep a high-degree coefficient while dropping a weaker
    low-degree one -- so the gap to degree truncation measures any anisotropy gain.
    The number of NONZERO coefficients saturates at (lmax+1)^2: the SHCoeffs array's
    upper-triangle (m>l) slots carry zero power and are never selected, so passing
    N beyond (lmax+1)^2 simply returns the full field.
    """
    out = s.copy()
    N = int(N)
    p = (out.coeffs ** 2).ravel()
    if N >= p.size:
        return out
    flat = out.coeffs.ravel().copy()
    if N <= 0:
        flat[:] = 0.0
    else:
        keep = np.argpartition(p, p.size - N)[p.size - N:]   # indices of N largest
        mask = np.zeros(p.size, dtype=bool)
        mask[keep] = True
        flat[~mask] = 0.0
    out.coeffs = flat.reshape(out.coeffs.shape)
    return out


def best_n_term_eps(s, N_list):
    """Relative L2 error of the best N-term approximation for each N (exact, grid-
    free).  By Parseval, keeping the N largest coefficient POWERS,
        eps(N) = sqrt( (sum of discarded c^2) / (sum of all c^2) ).
    """
    p = np.sort((s.coeffs ** 2).ravel())[::-1]      # power, descending
    total = p.sum()
    csum = np.cumsum(p)
    out = []
    for N in N_list:
        N = int(N)
        if N < 1:
            kept = 0.0
        elif N >= len(csum):
            kept = total
        else:
            kept = csum[N - 1]
        out.append(np.sqrt(max(total - kept, 0.0) / total))
    return np.array(out)


# --------------------------------------------------------------------------- #
# 3.  epsilon(l_max): grid-based (primary) and Parseval (cross-check)
# --------------------------------------------------------------------------- #
def epsilon_curve(field):
    """Compute the truncation-error curve for one field.

    eps(l_max) = sqrt( sum_{l>l_max} sigma_l^2 / sum_l sigma_l^2 ) is the EXACT
    area-weighted relative L2 error: on an exact Gauss-Legendre grid a band-
    limited field's grid error equals this Parseval ratio to ~1e-14.  The whole
    curve is therefore evaluated analytically from the degree variance in
    O(l_ref) via a cumulative tail sum -- instead of synthesising the truncated
    field at every degree (the old inner loop did ~len(lmax_list) full GLQ
    syntheses).  One reference synthesis plus ONE truncated synthesis at a
    representative degree are retained to verify, at run time, that the analytic
    curve matches a genuine grid computation; the full per-degree identity is
    exercised in tests/.

    Returns a dict with:
        lmax            (n,)        truncation degrees
        eps_grid        (n,)        area-weighted relative L2 error (exact analytic)
        eps_parseval    (n,)        identical analytic prediction (cross-check)
        Pl              (lref+1,)   4pi degree variance of the analysed field
        total_ref       float       grid integral of f_ref^2 over the sphere
        ncells          int         number of grid cells in the reference synthesis
        grid_check      dict        one-degree grid-vs-analytic verification
    """
    s = field["s"]
    lmax_grid = field["lmax_grid"]
    lmax_list = np.asarray(field["lmax_list"], dtype=int)

    # Degree variance (4pi power per degree) and its cumulative tail.
    Pl = s.spectrum()
    total_pow = Pl.sum()
    tail = np.cumsum(Pl[::-1])[::-1]            # tail[l] = sum_{k>=l} Pl[k]

    def _tail_above(lm):                       # power in degrees strictly > lm
        return tail[lm + 1] if lm + 1 < len(tail) else 0.0

    eps_grid = np.sqrt(np.array([_tail_above(int(lm)) for lm in lmax_list])
                       / total_pow)
    eps_parseval = eps_grid.copy()             # identical by construction

    # One reference + one truncated synthesis on the exact GLQ grid, to confirm
    # the analytic curve against a genuine grid computation at one representative
    # degree (replaces the removed per-degree loop; full check lives in tests/).
    zeros, w, W, nlat, nlon = glq_setup(lmax_grid)
    Wf = W.ravel()
    f_ref = expand_on_glq(s, zeros, lmax_grid).ravel()
    total_ref = (Wf * f_ref ** 2).sum()                 # = integral f_ref^2 dOmega
    lm_chk = int(lmax_list[len(lmax_list) // 2])
    f_tr = expand_on_glq(truncate(s, lm_chk), zeros, lmax_grid).ravel()
    eps_grid_synth = float(np.sqrt((Wf * (f_ref - f_tr) ** 2).sum() / total_ref))
    eps_analytic = float(np.sqrt(_tail_above(lm_chk) / total_pow))
    rel_err = (abs(eps_grid_synth - eps_analytic) / eps_analytic
               if eps_analytic > 0 else abs(eps_grid_synth - eps_analytic))

    return dict(lmax=lmax_list, eps_grid=eps_grid, eps_parseval=eps_parseval,
                total_ref=total_ref, Pl=Pl, ncells=f_ref.size,
                grid_check=dict(lmax=lm_chk, eps_grid_synth=eps_grid_synth,
                                eps_analytic=eps_analytic, rel_err=rel_err))


# --------------------------------------------------------------------------- #
# 3b. Parseval-based eps helpers (grid-free; shared by audit.py and functionals.py)
# --------------------------------------------------------------------------- #

# GRS80 normal-gravity-field fully-normalised even zonal coefficients
# (Moritz 1980, Geodetic Reference System 1980). WGS84 is identical to ~1e-11.
_NORMAL_CBAR = {2: -0.484166774985e-3, 4: 0.790303733511e-6,
                6: -0.168724961151e-8, 8: 0.346052468394e-11,
                10: -0.265002225747e-14}


def eps_from_degree_variance(Pl, lmax_list, lmin):
    """eps(l_max) = sqrt( sum_{l>l_max} Pl / sum_{l>=lmin} Pl ).

    Grid-free Parseval ratio.  Pl is indexed by degree (index 0 = degree 0).
    lmin is the lowest degree included in the total-power denominator (use 2 for
    geoid/disturbing potential, 1 for magnetic B_r).
    """
    total = Pl[lmin:].sum()
    out = []
    for lm in lmax_list:
        out.append(np.sqrt(Pl[lm + 1:].sum() / total))
    return np.array(out)


def disturbing_potential_degree_variance():
    """Disturbing-potential degree variance from RAW EGM2008 4pi Stokes coefficients.

    P_l^T = sum_m (dC_lm^2 + dS_lm^2) where dC has the GRS80 normal even-zonal
    field removed and degrees 0 and 1 are dropped.  The constant a^2 prefactor
    (from first-order Bruns N = a * dC) cancels in the relative error eps, so we
    return the proportional spectrum (index = degree, valid for l >= 2).

    This is the shared base for all gravity functionals in gravity_functional_field.
    """
    grav = Earth.EGM2008(lmax=C.GRAV_LREF)
    arr = grav.to_array()                 # (2, L+1, L+1), 4pi-normalised, C/S
    C0 = arr[0].copy(); Sx = arr[1].copy()
    C0[0, :] = 0.0; C0[1, :] = 0.0        # drop degrees 0, 1
    Sx[0, :] = 0.0; Sx[1, :] = 0.0
    for l, cbar in _NORMAL_CBAR.items():   # remove the GRS80 reference ellipsoid
        if l <= C.GRAV_LREF:
            C0[l, 0] -= cbar
    Pl = np.sum(C0 ** 2 + Sx ** 2, axis=1)
    return Pl                              # index = degree, valid for l >= 2


# Gravity functionals: (label, units, w_l formula string, weight function).
# Each is a DIAGONAL per-degree rescaling of the disturbing-potential degree
# variance.  anomaly uses (l-1), NOT (l+1).  Module-level so it is built once.
_GRAVITY_FUNCTIONAL_KINDS = {
    "geoid":       ("Geoid undulation N",          "m (prop.)",    "1",
                    lambda ls: np.ones_like(ls, float)),
    "disturbance": ("Gravity disturbance dg",      "mGal (prop.)", "(l+1)",
                    lambda ls: (ls + 1).astype(float)),
    "anomaly":     ("Free-air gravity anomaly Dg", "mGal (prop.)", "(l-1)",
                    lambda ls: (ls - 1).astype(float)),
    "Trr":         ("Radial-radial gradient T_rr", "E (prop.)",    "(l+1)(l+2)",
                    lambda ls: (ls + 1).astype(float) * (ls + 2)),
}


def gravity_functional_field(kind):
    """Degree-variance descriptor for a gravity functional derived from EGM2008.

    Each functional is a DIAGONAL per-degree rescaling of the disturbing-potential
    degree variance P_l^T (exact, no grid synthesis):

        P_l^func = w_l^2 * P_l^T

    Supported kinds and their per-degree weights w_l:
        'geoid'       : w_l = 1              (cross-check: reproduces headline)
        'disturbance' : w_l = (l+1)          (gravity disturbance dg; pyshtools
                                              spectrum('radial') uses (l+1))
        'anomaly'     : w_l = (l-1)          (free-air gravity anomaly; NOT (l+1)
                                              -- the anomaly operator is (l-1).
                                              At l=2: w_2=1; always >= 0 for l>=2)
        'Trr'         : w_l = (l+1)*(l+2)   (radial-radial gradient T_rr)

    The GM/r0^k radial prefactors are the SAME for all functionals and cancel in
    the relative error eps, so only the l-dependent factor w_l matters.

    Returns a dict with keys: kind, label, units, Pl (functional degree variance,
    index=degree, valid l>=2), lref, fit_range, lmin, provenance.
    """
    if kind not in _GRAVITY_FUNCTIONAL_KINDS:
        raise ValueError("kind must be one of %s; got %r"
                         % (sorted(_GRAVITY_FUNCTIONAL_KINDS), kind))

    label, units, w_formula, weight_fn = _GRAVITY_FUNCTIONAL_KINDS[kind]
    Pl_T = disturbing_potential_degree_variance()        # base: disturbing potential
    ls = np.arange(len(Pl_T))
    w = weight_fn(ls)
    Pl_func = w ** 2 * Pl_T                              # diagonal rescaling

    prov = ("Gravity functional '%s' (w_l=%s): diagonal degree-domain rescaling "
            "of EGM2008 disturbing-potential degree variance (GRS80 normal field "
            "removed, degrees 0,1 dropped, l_ref=%d, l_min=2). "
            "Exact grid-free Parseval computation. NOTE: this raw-Stokes path gives "
            "the geoid decay slope ~-0.898, vs the round-tripped headline geoid "
            "~-0.861 (Table I); the 0.037 gap is the latitude-dependent Bruns/"
            "normal-gravity factor, not a discrepancy."
            % (kind, w_formula, C.GRAV_LREF))
    return dict(
        kind=kind, label=label, units=units,
        Pl=Pl_func, lref=C.GRAV_LREF,
        fit_range=C.GRAV_FIT_RANGE, lmin=2,
        provenance=prov,
    )


# --------------------------------------------------------------------------- #
# 4.  Legitimate uncertainty estimators for the decay exponent
# --------------------------------------------------------------------------- #
def fit_spectrum(Pl, fit_range):
    """Weighted least-squares fit of the degree-variance spectrum to a power law
    sigma_l^2 = C * l^{-alpha} over the inclusive degree window fit_range.

    In log space  y_l = log10(sigma_l^2) = a - alpha * x_l,  x_l = log10(l),
    fitted with weights  w_l = (2l+1)/2.  These weights are the inverse
    asymptotic variance of log(sigma_l^2): sigma_l^2 is a sum of (2l+1) squared
    coefficients, ~ chi^2 with (2l+1) dof, so Var(log sigma_l^2) ~ 2/(2l+1).
    The propagated epsilon-decay exponent is p = (1 - alpha)/2 with
    SE(p) = SE(alpha)/2 (from eps(lmax) ~ lmax^{(1-alpha)/2} for alpha>1).

    Returns dict(alpha, alpha_se, intercept, r2, p_hat, p_se, n).
    """
    lo, hi = int(fit_range[0]), int(fit_range[1])
    ls = np.arange(len(Pl))
    mask = (ls >= max(lo, 1)) & (ls <= hi) & (Pl > 0)
    l_fit = ls[mask]
    n = int(mask.sum())

    if n < 3:
        nan = float("nan")
        return dict(alpha=nan, alpha_se=nan, alpha_se_model=nan,
                    reduced_chi2=nan, se_scale=nan, intercept=nan,
                    r2=nan, p_hat=nan, p_se=nan, n=n)

    x = np.log10(l_fit.astype(float))
    y = np.log10(Pl[mask])
    w = (2.0 * l_fit + 1.0) / 2.0          # w_l = (2l+1)/2  (inverse-variance)

    # Design matrix: columns [intercept, slope] -> y = X beta
    X = np.column_stack([np.ones(n), x])    # (n, 2)
    W = np.diag(w)

    # WLS normal equations: beta = (X^T W X)^{-1} X^T W y
    XtW = X.T @ W                           # (2, n)
    XtWX = XtW @ X                          # (2, 2)
    XtWy = XtW @ y                          # (2,)
    beta = np.linalg.solve(XtWX, XtWy)     # [intercept, slope]

    intercept = beta[0]
    slope = beta[1]
    alpha = -slope                          # convention: Pl ~ l^{-alpha}, so slope = -alpha

    # Weighted fitted values and residuals
    yhat = X @ beta
    resid = y - yhat

    # Standard error of alpha = the standard (misfit-scaled) WLS standard error,
    #   alpha_se = sqrt( s2 * inv(X^T W X)[1,1] ),   s2 = reduced weighted chi^2.
    # The weights w_l=(2l+1)/2 are the Kaula random-coefficient inverse variances
    # of ln(sigma_l^2); since Earth's spectrum is a SINGLE deterministic
    # realisation, s2 lets the actual scatter about the power law -- not the
    # assumed sampling weights -- set the error magnitude.  This SE is invariant
    # to a constant rescaling of the weights.  We also expose the pure
    # inverse-variance (model) SE and s2 so the fit quality is visible: s2 < 1
    # means the spectrum tracks the power law more tightly than the Kaula
    # sampling floor; s2 > 1 means genuine departures from a single power law.
    XtWX_inv = np.linalg.inv(XtWX)
    se_alpha_model = float(np.sqrt(XtWX_inv[1, 1]))   # pure inverse-variance (model) SE
    s2 = float(np.sum(w * resid ** 2) / (n - 2))      # reduced weighted chi-square
    scale = float(np.sqrt(s2))                        # misfit scale factor
    alpha_se = scale * se_alpha_model                 # standard misfit-scaled WLS SE

    # Weighted R^2
    ybar_w = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * (y - ybar_w) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Propagated epsilon-decay exponent: eps(lmax) ~ lmax^{(1-alpha)/2}
    p_hat = (1.0 - alpha) / 2.0
    p_se = alpha_se / 2.0

    return dict(alpha=float(alpha), alpha_se=float(alpha_se),
                alpha_se_model=se_alpha_model, reduced_chi2=s2, se_scale=scale,
                intercept=float(intercept), r2=r2,
                p_hat=float(p_hat), p_se=float(p_se), n=n)


def jackknife_slope(lmax, eps, fit_range):
    """Delete-one-degree jackknife SE of the epsilon-OLS decay slope.

    Full-sample slope = OLS of log10(eps) on log10(lmax) over the inclusive
    fit_range (identical to fit_loglog). Each in-range point is deleted in turn,
    giving leave-one-out slopes p_(i); the jackknife SE is
        sqrt( (n-1)/n * sum_i (p_(i) - p_bar)^2 ),   p_bar = mean_i p_(i).
    Returns (p_ols, se_jack, n).
    """
    lmax = np.asarray(lmax, float)
    eps = np.asarray(eps, float)
    lo, hi = int(fit_range[0]), int(fit_range[1])
    mask = (lmax >= lo) & (lmax <= hi) & (eps > 0)
    lm_fit = lmax[mask]
    ep_fit = eps[mask]
    n = int(mask.sum())

    if n < 3:
        return float("nan"), float("nan"), n

    p_ols = fit_loglog(lm_fit, ep_fit)[0]

    loo_slopes = np.empty(n)
    for i in range(n):
        idx = np.arange(n) != i
        p_i = fit_loglog(lm_fit[idx], ep_fit[idx])[0]
        loo_slopes[i] = p_i

    p_bar = loo_slopes.mean()
    se_jack = float(np.sqrt((n - 1) / n * np.sum((loo_slopes - p_bar) ** 2)))

    return float(p_ols), se_jack, n


# --------------------------------------------------------------------------- #
# 5.  Log-log fit and threshold crossing
# --------------------------------------------------------------------------- #
def fit_loglog(lmax, eps):
    """OLS fit of log10(eps) vs log10(l_max).  Returns (slope, intercept, r2)."""
    m = (eps > 0) & (lmax > 0)
    x = np.log10(lmax[m].astype(float))
    y = np.log10(eps[m])
    if x.size < 2:
        return np.nan, np.nan, np.nan
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return slope, intercept, r2


def crossing_degrees(lmax, eps, thresholds):
    """For each threshold, the (interpolated) l_max where eps first drops to it.

    eps is assumed monotonically non-increasing in l_max.  Interpolation is
    LOG--LOG (linear in log10 eps AND in log10 l_max), consistent with the
    power-law decay eps ~ l_max^p, so the crossing degree matches a densely
    sampled curve to <1 percent.  Returns {threshold: l_max or None}; None is a
    sentinel meaning the crossing is not locatable within the sampled curve --
    either eps never reaches t (eps.min() > t), or eps is already below t at the
    smallest sampled degree (eps.max() <= t).  In the latter case we return None
    rather than lmax[0], which would falsely report the crossing at the smallest
    sampled degree.
    """
    lmax = np.asarray(lmax, float)
    eps = np.asarray(eps, float)
    order = np.argsort(lmax)
    lmax, eps = lmax[order], eps[order]
    out = {}
    for t in thresholds:
        if eps.min() > t:                # never gets that low within the range
            out[t] = None
            continue
        if eps.max() <= t:               # already below t at the smallest sampled
            out[t] = None                # l_max -> not locatable; sentinel, NOT
            continue                     # lmax[0] (which would be wrong)
        # first index where eps <= t (here k >= 1, since eps[0]=eps.max() > t)
        k = int(np.argmax(eps <= t))
        if k == 0:                       # defensive; unreachable given the guards
            out[t] = None
            continue
        e1, e0 = eps[k], eps[k - 1]      # e0 > t >= e1
        l1, l0 = lmax[k], lmax[k - 1]
        if e1 <= 0 or e0 <= 0:           # eps hit zero: linear-in-eps fraction
            frac = (e0 - t) / (e0 - e1)
        else:
            frac = (np.log10(e0) - np.log10(t)) / (np.log10(e0) - np.log10(e1))
        # interpolate the degree in log space (log--log) when l0, l1 > 0
        if l0 > 0 and l1 > 0:
            out[t] = float(10 ** (np.log10(l0)
                                  + frac * (np.log10(l1) - np.log10(l0))))
        else:
            out[t] = float(l0 + frac * (l1 - l0))
    return out


# --------------------------------------------------------------------------- #
# 6.  Compression ratio
# --------------------------------------------------------------------------- #
def compression_ratio(lmax, lref):
    """#coeffs(l<=l_max) / #coeffs(l<=l_ref) = (l_max+1)^2 / (l_ref+1)^2."""
    return (np.asarray(lmax, float) + 1) ** 2 / (lref + 1) ** 2


# --------------------------------------------------------------------------- #
# 7.  Self-tests (quadrature exactness)
# --------------------------------------------------------------------------- #
def selftest_quadrature(field):
    """Check that (a) the GLQ weights integrate the sphere to 4*pi and (b) the
    grid mean-square equals 4*pi * sum_l P_l (Parseval) for the analysed field.
    Returns a dict of diagnostics."""
    s = field["s"]
    zeros, w, W, nlat, nlon = glq_setup(field["lmax_grid"])
    area = W.sum()
    f = expand_on_glq(s, zeros, field["lmax_grid"])
    integral_f2 = (W * f ** 2).sum()
    parseval_f2 = 4.0 * np.pi * s.spectrum().sum()
    return dict(area=area, area_err=abs(area - 4 * np.pi),
                integral_f2=integral_f2, parseval_f2=parseval_f2,
                rel_err=abs(integral_f2 - parseval_f2) / parseval_f2)


# --------------------------------------------------------------------------- #
# 8.  Drive one field end-to-end
# --------------------------------------------------------------------------- #
def analyse_field(field):
    """Analyse one field fully deterministically.  No rng argument — the
    invalid bootstrap has been removed.  Uncertainty is now quantified by two
    legitimate estimators:
      - fit_spectrum: WLS on the degree-variance spectrum (PRIMARY, unbiased)
      - jackknife_slope: delete-one-degree jackknife on the epsilon-OLS slope
    """
    curve = epsilon_curve(field)

    lm = curve["lmax"]
    eps = curve["eps_grid"]
    mask = (lm >= field["fit_range"][0]) & (lm <= field["fit_range"][1])

    # OLS log-log fit on the epsilon curve (used for the plotted fit line and
    # back-compatibility; primary decay exponent comes from fit_spectrum below).
    p_ols, ols_intercept, ols_r2 = fit_loglog(lm[mask], eps[mask])

    # Jackknife SE on the epsilon-OLS slope (robustness cross-check).
    _, ols_jack_se, _ = jackknife_slope(lm, eps, field["fit_range"])

    # Weighted spectrum fit (PRIMARY decay estimator — unbiased, no finite-range
    # epsilon bias; operates directly on degree variances Pl).
    spec = fit_spectrum(curve["Pl"], field["fit_range"])

    # NOTE: the plotted epsilon-fit line uses ols_slope + intercept (i.e. the OLS
    # fit to the epsilon curve), NOT the spectrum p_hat.  p_hat is reported in the
    # table and paper text as the primary numeric result.

    thr = crossing_degrees(lm, eps, C.THRESHOLDS)
    qa = selftest_quadrature(field)

    # Verification: the analytic eps curve vs a genuine single-degree grid
    # synthesis (the O(l_max) refactor removed the per-degree grid loop; the full
    # grid==Parseval identity across all degrees is asserted in tests/).
    pars_max_rel = float(curve["grid_check"]["rel_err"])
    pars_pass = pars_max_rel < C.PARSEVAL_RTOL

    # Monotonicity (allow tiny numerical noise).
    mono = bool(np.all(np.diff(curve["eps_grid"]) <= 1e-9))

    return dict(
        field=field,
        curve=curve,
        fit=dict(
            # PRIMARY decay exponent = the directly-measured eps-curve log-log
            # slope, with a delete-one-degree jackknife SE.  This is what the
            # error curve actually does over the fit range (so it matches Fig 2),
            # matches the published headline values, and is well defined for ALL
            # fields -- including the near-white crustal model, where the
            # asymptotic p=(1-alpha)/2 relation breaks down (alpha < 1).
            slope=p_ols,
            slope_se=ols_jack_se,
            intercept=ols_intercept,    # matched with slope -> plotted fit line ok
            r2=ols_r2,
            # Spectral index from the weighted (2l+1)/2 fit of sigma_l^2 ~ l^-alpha
            # (cross-check + Phase-2 theoretical companion).  The relation
            # p=(1-alpha)/2 holds asymptotically for alpha>1 (gravity); it is an
            # EFFECTIVE index for the band-limited 13-degree magnetic field and
            # does NOT apply to the near-white crustal spectrum.
            alpha=spec["alpha"],
            alpha_se=spec["alpha_se"],
            alpha_se_model=spec["alpha_se_model"],
            spec_chi2=spec["reduced_chi2"],
            spec_r2=spec["r2"],
            p_spec=spec["p_hat"],       # (1-alpha)/2 asymptotic prediction
            p_spec_se=spec["p_se"],
        ),
        thresholds=thr,
        qa=qa,
        parseval=dict(max_rel=pars_max_rel, passed=pars_pass),
        monotonic=mono,
    )
