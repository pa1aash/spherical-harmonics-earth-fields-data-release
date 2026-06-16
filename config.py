"""
Configuration for the spherical-harmonic spectral-truncation study.

Seed 1A: Spectral truncation of Earth's gravity (EGM2008) vs magnetic (IGRF-13)
fields.  All tunable parameters live here so the whole study is reproducible from
one place.  See README.md for the scientific rationale behind each choice.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
# The analysis is fully DETERMINISTIC.  eps(l_max) is the exact area-weighted
# Parseval ratio (no sampling), and the decay-exponent uncertainty is quantified
# analytically (weighted-least-squares spectrum fit) and by a delete-one-degree
# jackknife -- there is no random resampling step.  The earlier 30%-cell
# "bootstrap" was removed in Phase 1: subsampling a deterministic field does not
# produce a confidence interval.
RANDOM_SEED = 20240606          # retained for provenance only; no stochastic step remains

# --------------------------------------------------------------------------- #
# Reference (truncation degree) and grid resolutions
# --------------------------------------------------------------------------- #
# GRAVITY (EGM2008)
GRAV_LREF = 360                 # reference degree the "ground truth" is built to
GRAV_LMAX_GRID = 360            # GLQ analysis grid bandwidth (>= GRAV_LREF, exact)
GRAV_MAP_LMAX = 360             # DH2 map grid bandwidth (regular lat/lon, plotting)
GRAV_GEOID_ORDER = 1            # Bruns order; 1 => geoid is exactly band-limited
GRAV_MAP_DEGREES = [10, 30, 90, 360]   # Fig 1 reconstruction panels

# MAGNETIC (IGRF-13)
MAG_LREF = 13                   # IGRF-13 is intrinsically degree 13
MAG_YEAR = 2020.0               # IGRF-13 epoch
MAG_LMAX_GRID = 90              # oversampled GLQ analysis grid (field band-limited 13)
MAG_MAP_LMAX = 90               # DH2 map grid bandwidth

# Optional secondary (high-degree lithospheric) magnetic model for a longer curve.
INCLUDE_LITHOSPHERIC = True     # NGDC-720 V3 (Maus 2010); kept clearly separate
LITHO_LREF = 133                # truncate the (very large) crustal model here
LITHO_LMAX_GRID = 133

# --------------------------------------------------------------------------- #
# Truncation degrees sampled for the epsilon(l_max) curve
# --------------------------------------------------------------------------- #
def _logspaced_degrees(lref, extra=()):
    """Integer degrees ~log-spaced over [2, lref], unioned with `extra`.

    Dense enough (~48 points) that the epsilon and Pareto curves render as smooth
    lines rather than a few markers.
    """
    import numpy as np
    pts = np.unique(np.round(np.geomspace(2, lref, 48)).astype(int))
    pts = np.unique(np.concatenate([pts, np.asarray(list(extra), dtype=int),
                                    [lref]]))
    pts = pts[(pts >= 1) & (pts <= lref)]
    return pts.tolist()

GRAV_LMAX_LIST = _logspaced_degrees(GRAV_LREF, extra=GRAV_MAP_DEGREES)
MAG_LMAX_LIST = list(range(1, MAG_LREF + 1))           # 1..13
LITHO_LMAX_LIST = _logspaced_degrees(LITHO_LREF, extra=(16, 20, 30, 50, 80))

# --------------------------------------------------------------------------- #
# Log-log fit ranges for the empirical decay exponent (avoid the low-degree
# non-power-law head and the finite-reference roll-off near l_ref).
# --------------------------------------------------------------------------- #
GRAV_FIT_RANGE = (4, 300)       # inclusive l_max bounds for OLS fit
MAG_FIT_RANGE = (1, 12)
LITHO_FIT_RANGE = (16, 110)

# --------------------------------------------------------------------------- #
# Error thresholds to locate (epsilon crosses these values)
# --------------------------------------------------------------------------- #
THRESHOLDS = [0.10, 0.05, 0.01]        # 10%, 5%, 1% (locked primary outcome)
# Extra error levels marked on the storage/Pareto chart and tabulated alongside.
MARKER_THRESHOLDS = [0.15, 0.10, 0.05, 0.025, 0.01]   # 15/10/5/2.5/1 %

# --------------------------------------------------------------------------- #
# Verification tolerance: grid-based epsilon vs analytic Parseval prediction
# --------------------------------------------------------------------------- #
PARSEVAL_RTOL = 1e-3            # relative tolerance for PASS/FAIL

# --------------------------------------------------------------------------- #
# Plotting: colourblind-friendly palette (Wong 2011, Nature Methods)
# --------------------------------------------------------------------------- #
COLOR_GRAVITY = "#0072B2"      # blue
COLOR_MAGNETIC = "#D55E00"     # vermillion
COLOR_LITHO = "#009E73"        # bluish green
COLOR_FIT = "#000000"          # black dashed fit line
COLOR_THRESH = "#555555"       # grey threshold guides
DIVERGING_CMAP = "RdBu_r"      # diverging, ~colourblind-safe, centred at zero

DPI = 300
FIG_FORMATS = ("png", "pdf")   # raster + vector

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_FIGURES = os.path.join(ROOT, "figures")
DIR_RESULTS = os.path.join(ROOT, "results")
DIR_DATA = os.path.join(ROOT, "data")
