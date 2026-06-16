"""
test_radius.py  --  TDD for radius.py (Component 5: radius-dependence of
                    magnetic-field compressibility).

ALL tests use SYNTHETIC Gauss-coefficient arrays (no pyshtools Earth dataset
downloads).  A fixed-seed random red spectrum (g, h ~ Normal(0, sigma_l))
with sigma_l^2 ~ (l+1)^{-5} is used so the surface field has a strongly red
spectrum (high compressibility), providing a clear monotone reversal when
continued downward.

Three invariants tested
-----------------------
1. At r = a (surface), scaled spectrum == unscaled spectrum  (a/r = 1 -> factor 1^(l+1) = 1).
2. As r decreases toward the CMB, l@1% is NON-DECREASING -- downward continuation
   un-reddens the spectrum, reducing compressibility (field becomes harder to represent
   compactly as we approach the source).
3. As r decreases, the fraction of total power above degree 6 INCREASES -- un-reddening
   shifts power toward high degrees.
"""
from __future__ import annotations

import numpy as np
import pytest

from radius import (
    scale_gauss_array,
    Br_degree_variance_from_array,
    radius_sweep,
    A_KM,
    CMB_KM,
    LMAX,
    LMIN,
    FIT_RANGE,
    THRESH_1PCT,
)
import shtrunc as S


# --------------------------------------------------------------------------- #
# Synthetic Gauss-coefficient array -- no dataset downloads
# --------------------------------------------------------------------------- #
def _make_synthetic_arr(lmax=13, seed=42):
    """Random (seeded) Schmidt Gauss-coefficient array with a red power profile.

    sigma_l^2 ~ (l+1)^{-5}  (strongly red -- dipole-dominated like IGRF).
    Shape: (2, lmax+1, lmax+1), index 0 = g, index 1 = h, degree-0 set to 0.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(1, lmax + 1):
        sigma = (l + 1) ** (-2.5)          # sigma_l^2 = sigma^2 = (l+1)^{-5}
        for m in range(l + 1):
            arr[0, l, m] = rng.standard_normal() * sigma   # g_lm
            if m > 0:
                arr[1, l, m] = rng.standard_normal() * sigma  # h_lm (m>0 only)
    return arr


_ARR = _make_synthetic_arr(lmax=LMAX, seed=42)
_A = A_KM


# --------------------------------------------------------------------------- #
# Test 1: at r = a the scaled array equals the unscaled array
# --------------------------------------------------------------------------- #
class TestSurfaceScalingIsIdentity:
    """scale_gauss_array(arr, a, r=a, lmax) must return arr unchanged."""

    def test_surface_scale_identity(self):
        scaled = scale_gauss_array(_ARR, _A, _A, LMAX)
        np.testing.assert_allclose(
            scaled, _ARR, rtol=0.0, atol=1e-14,
            err_msg="Scaling at r=a must be the identity; max deviation = %.3e"
                    % float(np.max(np.abs(scaled - _ARR)))
        )

    def test_surface_Pl_unchanged(self):
        """The degree variance at r=a must equal the unscaled degree variance."""
        Pl_surf = Br_degree_variance_from_array(
            scale_gauss_array(_ARR, _A, _A, LMAX), LMAX
        )
        Pl_unscaled = Br_degree_variance_from_array(_ARR, LMAX)
        np.testing.assert_allclose(
            Pl_surf, Pl_unscaled, rtol=1e-13,
            err_msg="Degree variance at r=a must match unscaled."
        )


# --------------------------------------------------------------------------- #
# Test 2: l@1% is non-decreasing as r decreases toward the CMB
# --------------------------------------------------------------------------- #
class TestMonotoneUnreddening:
    """Downward continuation must make the field monotonically less compressible.

    l@1% is non-decreasing as r decreases (spectrum flattens -> harder to compress).
    We use a dense 10-point sweep from surface to CMB.
    """

    def _sweep(self):
        # Surface first (largest r), descending to CMB (smallest r).
        # np.linspace(A, CMB) goes surface->CMB; [::-1] reverses to CMB->surface.
        # We want surface-first, so use linspace without reversal then flip.
        r_values = np.linspace(CMB_KM, _A, 10)[::-1]   # surface first
        lmax_list = list(range(1, LMAX + 1))
        rows = radius_sweep(_ARR, _A, r_values, LMAX, lmax_list,
                            FIT_RANGE, THRESH_1PCT)
        return rows

    def test_l_at_1pct_nondecreasing(self):
        """l@1% must not decrease as r goes from surface to CMB."""
        rows = self._sweep()
        # Collect the l@1% values; skip None (threshold not bracketed at deep radii)
        l_vals = [row["l_at_thresh"] for row in rows
                  if row["l_at_thresh"] is not None]

        assert len(l_vals) >= 2, (
            "Need at least 2 bracketed l@1% values to test monotonicity; "
            "got %d.  Spectrum may be too flat everywhere." % len(l_vals)
        )
        for i in range(len(l_vals) - 1):
            assert l_vals[i + 1] >= l_vals[i] - 1e-9, (
                "l@1%% decreased from %.3f (row %d) to %.3f (row %d); "
                "expected non-decreasing." % (l_vals[i], i, l_vals[i + 1], i + 1)
            )

    def test_l_at_1pct_strictly_increases(self):
        """The FIRST and LAST bracketed l@1% values must show a clear increase.

        Checks that the un-reddening is not trivially zero, i.e. the synthetic
        spectrum is red enough at the surface that downward continuation
        meaningfully shifts l@1%.
        """
        rows = self._sweep()
        l_vals = [row["l_at_thresh"] for row in rows
                  if row["l_at_thresh"] is not None]
        # It is acceptable if the CMB spectrum is too flat to bracket the 1%
        # threshold (l@1% goes off the top of the 13-degree band); in that case
        # we assert the first bracketed value is smaller than LMAX-1, confirming
        # the surface is compressible and the un-reddening effect is present.
        if len(l_vals) >= 2:
            assert l_vals[-1] > l_vals[0], (
                "Expected l@1%% to be strictly larger deep than at surface; "
                "surface=%.3f  deepest-bracketed=%.3f." % (l_vals[0], l_vals[-1])
            )
        else:
            # Only surface is bracketed -> CMB has gone off the scale (un-reddened
            # beyond degree 13), which is the strongest form of the effect.
            assert l_vals[0] < LMAX - 0.5, (
                "Surface l@1%%=%.3f is already near lmax=%d; spectrum not red "
                "enough to show un-reddening." % (l_vals[0], LMAX)
            )


# --------------------------------------------------------------------------- #
# Test 3: power fraction above degree 6 increases as r decreases
# --------------------------------------------------------------------------- #
class TestHighDegreePowerIncreases:
    """Downward continuation must shift power toward high degrees.

    Fraction of B_r power above degree 6 must increase monotonically as r
    decreases from surface to CMB.
    """

    _SPLIT = 6     # power above this degree (degrees 7..lmax)

    def _high_degree_fraction(self, arr, r):
        """Fraction of total B_r power in degrees > _SPLIT at radius r."""
        scaled = scale_gauss_array(arr, _A, r, LMAX)
        Pl = Br_degree_variance_from_array(scaled, LMAX)
        total = Pl[LMIN:].sum()
        high = Pl[self._SPLIT + 1:].sum()
        return float(high / total) if total > 0 else 0.0

    def test_high_degree_fraction_increases(self):
        """Fraction of power above degree 6 must increase from surface to CMB."""
        r_values = np.linspace(CMB_KM, _A, 10)[::-1]   # surface first (largest r)
        fractions = [self._high_degree_fraction(_ARR, r) for r in r_values]

        for i in range(len(fractions) - 1):
            # Allow tiny numerical noise with a 1e-12 tolerance
            assert fractions[i + 1] >= fractions[i] - 1e-12, (
                "High-degree power fraction decreased from r=%.1f (frac=%.6f) "
                "to r=%.1f (frac=%.6f); expected non-decreasing."
                % (r_values[i], fractions[i], r_values[i + 1], fractions[i + 1])
            )

        # The total increase must be substantial (un-reddening is not trivial)
        assert fractions[-1] > fractions[0] + 1e-6, (
            "Expected a substantial increase in high-degree fraction from "
            "surface (%.6f) to CMB (%.6f)." % (fractions[0], fractions[-1])
        )


# --------------------------------------------------------------------------- #
# Additional unit test: Br_degree_variance_from_array degree-0 is zero
# --------------------------------------------------------------------------- #
class TestDegreeVarianceProperties:
    """Small sanity checks on Br_degree_variance_from_array."""

    def test_degree_zero_is_zero(self):
        """Degree-0 degree variance must be zero (no magnetic monopole)."""
        Pl = Br_degree_variance_from_array(_ARR, LMAX)
        assert Pl[0] == 0.0, "Pl[0] should be 0 (no monopole); got %.6e" % Pl[0]

    def test_positive_for_nonzero_degrees(self):
        """Degree variances for l >= 1 must be strictly positive with non-zero input."""
        Pl = Br_degree_variance_from_array(_ARR, LMAX)
        for l in range(1, LMAX + 1):
            assert Pl[l] > 0.0, "Pl[%d] = %.6e, expected > 0." % (l, Pl[l])

    def test_scaling_increases_high_degree_power(self):
        """(a/r)^(l+1) with r < a must increase high-degree relative to low."""
        r = CMB_KM   # strong downward continuation
        scaled = scale_gauss_array(_ARR, _A, r, LMAX)
        Pl_surf = Br_degree_variance_from_array(_ARR, LMAX)
        Pl_cmb = Br_degree_variance_from_array(scaled, LMAX)
        # l=13 amplification = (a/r)^14 which is (6371.2/3480)^14 ~ 2.63e4
        ratio_l13 = float(Pl_cmb[13] / Pl_surf[13]) if Pl_surf[13] > 0 else 0.0
        ratio_l1 = float(Pl_cmb[1] / Pl_surf[1]) if Pl_surf[1] > 0 else 0.0
        assert ratio_l13 > ratio_l1, (
            "Degree-13 amplification (%.2e) should exceed degree-1 (%.2e)."
            % (ratio_l13, ratio_l1)
        )
