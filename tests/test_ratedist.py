"""test_ratedist.py  --  pytest suite for ratedist.py (rate-distortion floor).

All tests operate on SYNTHETIC spectra -- no network access, no dataset
downloads, no pyshtools Earth datasets.

Synthetic spectrum: Pl = (l+1)^{-3} for l = 0..40, lmin = 0.
(The lmin=0 choice keeps the test self-contained; the public API accepts any lmin.)
"""
from __future__ import annotations

import numpy as np
import pytest

from ratedist import _D_of_mu, _bisect_mu, _rate_distortion, water_fill_spectrum


# --------------------------------------------------------------------------- #
# Shared synthetic spectrum
# --------------------------------------------------------------------------- #
L = 40
LS = np.arange(L + 1)
PL_SYNTH = (LS + 1.0) ** (-3.0)    # Pl = (l+1)^{-3} for l = 0..40
LMIN = 0                            # include l=0 so the test is fully self-contained

# Pre-compute the quantities used across tests
_COUNTS = (2 * LS + 1).astype(float)
_SIGMA2 = PL_SYNTH / _COUNTS
_TOTAL_VAR = float(PL_SYNTH[LMIN:].sum())
_N_TOTAL = int(_COUNTS[LMIN:].sum())


# --------------------------------------------------------------------------- #
# Test 1 — At the solved mu, D(mu) == D_abs to < 1e-9
# --------------------------------------------------------------------------- #
class TestBisectionAccuracy:
    """The bisection must find mu* such that D(mu*) == D_abs within tolerance."""

    @pytest.mark.parametrize("eps", [0.10, 0.05, 0.025, 0.01, 0.001])
    def test_distortion_target_hit(self, eps):
        """D(mu*) equals D_abs to < 1e-9 for each tested eps."""
        result = _rate_distortion(PL_SYNTH, eps, LMIN)
        mu = result["mu"]
        D_abs = result["D_abs"]
        sigma2 = _SIGMA2[LMIN:]
        counts = _COUNTS[LMIN:]
        ls = LS[LMIN:]

        D_achieved = _D_of_mu(mu, sigma2, counts)
        assert abs(D_achieved - D_abs) < 1e-9, (
            "eps=%.4f: D(mu*)=%.15e, D_abs=%.15e, |diff|=%.3e"
            % (eps, D_achieved, D_abs, abs(D_achieved - D_abs))
        )


# --------------------------------------------------------------------------- #
# Test 2 — B_WF decreases monotonically as eps increases
# --------------------------------------------------------------------------- #
class TestRateMonotone:
    """Coarser targets (larger eps) require fewer bits (smaller B_WF)."""

    def test_rate_monotone_decreasing(self):
        """B_WF must be strictly decreasing as eps increases."""
        eps_list = [0.01, 0.025, 0.05, 0.10]     # ascending -> B_WF descending
        rates = [_rate_distortion(PL_SYNTH, eps, LMIN)["B_WF_bits"]
                 for eps in eps_list]
        for i in range(len(rates) - 1):
            assert rates[i] > rates[i + 1], (
                "B_WF NOT decreasing: eps=%.3f -> %.3f bits, "
                "eps=%.3f -> %.3f bits"
                % (eps_list[i], rates[i], eps_list[i + 1], rates[i + 1])
            )


# --------------------------------------------------------------------------- #
# Test 3 — B_WF >= 0; B_WF == 0 (or near 0) when eps^2 >= 1
# --------------------------------------------------------------------------- #
class TestRateNonNegativeAndZeroAtFullDistortion:
    """B_WF must be non-negative; at eps >= 1 the rate floor is zero."""

    @pytest.mark.parametrize("eps", [0.10, 0.05, 0.025, 0.01, 0.001])
    def test_nonnegative(self, eps):
        """B_WF >= 0 for any eps in (0, 1)."""
        result = _rate_distortion(PL_SYNTH, eps, LMIN)
        assert result["B_WF_bits"] >= 0.0, (
            "eps=%.4f: B_WF=%.6f is negative" % (eps, result["B_WF_bits"])
        )

    def test_zero_at_full_distortion(self):
        """At eps = 1 all distortion is allowed; B_WF should be 0 (or near 0).

        eps^2 = 1 means D_abs = total_var, so the water level mu >= max(sigma2)
        and every dimension is 'flooded' (zero rate allocated).
        """
        result = _rate_distortion(PL_SYNTH, 1.0, LMIN)
        assert result["B_WF_bits"] < 1e-6, (
            "eps=1.0: expected B_WF ~ 0, got %.6f bits" % result["B_WF_bits"]
        )

    def test_zero_beyond_full_distortion(self):
        """At eps > 1 the distortion budget exceeds total variance; B_WF == 0."""
        result = _rate_distortion(PL_SYNTH, 2.0, LMIN)
        assert result["B_WF_bits"] < 1e-6, (
            "eps=2.0: expected B_WF ~ 0, got %.6f bits" % result["B_WF_bits"]
        )


# --------------------------------------------------------------------------- #
# Test 4 — White spectrum closed-form check
# --------------------------------------------------------------------------- #
class TestWhiteSpectrumClosedForm:
    """For a white spectrum (all sigma^2_l equal), the water-filling solution
    has a closed form.

    If sigma^2_l = sigma^2 for all l, then:
        D(mu) = N_total * min(sigma^2, mu)
    The unique solution for D(mu*) = D_abs = eps^2 * N_total * sigma^2 is:
        mu* = eps^2 * sigma^2
    And the rate is:
        B_WF = (N/2) * log2(sigma^2 / mu*) = (N/2) * log2(1/eps^2)
             = -N * log2(eps)
    where N = N_total.
    """

    # Build a white spectrum: all per-coefficient variances equal.
    # Use l=1..20 so counts = 3,5,...,41 (non-uniform) but sigma^2 is uniform.
    _LS_W = np.arange(1, 21)
    _COUNTS_W = (2 * _LS_W + 1).astype(float)
    _SIGMA2_CONST = 1e-4
    # Pl_white = sigma^2_const * (2l+1) so that Pl/(2l+1) = sigma^2_const
    _PL_WHITE = _SIGMA2_CONST * _COUNTS_W
    # Pad to full array indexed by degree (index 0 = degree 0 = 0)
    _PL_WHITE_FULL = np.zeros(21)
    _PL_WHITE_FULL[1:] = _PL_WHITE
    _LMIN_W = 1
    _N_W = int(_COUNTS_W.sum())

    @pytest.mark.parametrize("eps", [0.10, 0.05, 0.025, 0.01])
    def test_white_closed_form(self, eps):
        """B_WF matches (N/2)*log2(1/eps^2) for a white spectrum."""
        N = self._N_W
        B_expected = (N / 2.0) * np.log2(1.0 / (eps ** 2))

        result = _rate_distortion(self._PL_WHITE_FULL, eps, self._LMIN_W)
        B_WF = result["B_WF_bits"]

        # Tolerance: floating-point bisection at ~1e-14 relative -> B_WF accurate
        # to many significant figures; allow 1e-6 absolute tolerance.
        assert abs(B_WF - B_expected) < 1e-6, (
            "White spectrum eps=%.4f: B_WF=%.10f, expected=%.10f, diff=%.3e"
            % (eps, B_WF, B_expected, abs(B_WF - B_expected))
        )

    @pytest.mark.parametrize("eps", [0.10, 0.05, 0.025, 0.01])
    def test_white_mu_closed_form(self, eps):
        """mu* = eps^2 * sigma^2 for a white spectrum.

        The bisection converges at the level of the distortion target D_abs
        (tolerance 1e-14 * D_abs in _bisect_mu), not at the level of mu itself.
        The resulting relative error in mu can be as large as ~1e-12 because mu
        appears INSIDE a sum over many degrees (the gradient of D w.r.t. mu is
        N_total, so a relative mu error of ~tol/N_total arises naturally).
        We therefore check mu to 1e-10 relative, consistent with the 1e-9
        distortion tolerance required by Test 1.
        """
        mu_expected = (eps ** 2) * self._SIGMA2_CONST
        result = _rate_distortion(self._PL_WHITE_FULL, eps, self._LMIN_W)
        mu = result["mu"]
        rel_diff = abs(mu - mu_expected) / mu_expected
        assert rel_diff < 1e-10, (
            "White spectrum eps=%.4f: mu=%.15e, expected=%.15e, rel_diff=%.3e"
            % (eps, mu, mu_expected, rel_diff)
        )


# --------------------------------------------------------------------------- #
# Test 5 — water_fill_spectrum public API sanity
# --------------------------------------------------------------------------- #
class TestPublicAPI:
    """water_fill_spectrum must return one dict per eps target with the right keys."""

    _EPS_LIST = [0.10, 0.05, 0.025, 0.01]
    _REQUIRED_KEYS = {"eps", "mu", "B_WF_bits", "bits_per_coeff",
                      "l_trunc", "n_coeffs_trunc", "B_float32_bits",
                      "float32_over_floor_ratio"}

    def test_output_length(self):
        rows = water_fill_spectrum(PL_SYNTH, self._EPS_LIST, LMIN)
        assert len(rows) == len(self._EPS_LIST)

    def test_required_keys(self):
        rows = water_fill_spectrum(PL_SYNTH, self._EPS_LIST, LMIN)
        for r in rows:
            missing = self._REQUIRED_KEYS - set(r.keys())
            assert not missing, "Missing keys: %s" % missing

    def test_ratio_nonnegative_when_defined(self):
        rows = water_fill_spectrum(PL_SYNTH, self._EPS_LIST, LMIN)
        for r in rows:
            ratio = r["float32_over_floor_ratio"]
            if ratio is not None and not (ratio == float("inf")):
                assert ratio >= 0.0, "Negative ratio: %s" % ratio
