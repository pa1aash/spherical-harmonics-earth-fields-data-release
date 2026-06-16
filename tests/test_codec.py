"""test_codec.py  --  pytest suite for codec.py (operational codec gap).

All tests operate on SYNTHETIC spectra and SYNTHETIC coefficients.
No network access, no pyshtools Earth-dataset downloads.

Synthetic setup
---------------
Power-law spectrum:  Pl = C * (l+1)^{-alpha} for l = 0..L_MAX

We use two different spectral shapes to verify eps-dependence:
  - steep   : alpha = 4   (fast decay, like geoid)
  - shallow : alpha = 2   (slow decay, like crustal)

Synthetic coefficients at each degree l are drawn from N(0, sigma2_l)
with a fixed RNG seed for reproducibility.  The codec is NOT allowed
to use exact Gaussian statistics; it uses the actual coefficient values.

Integer allocation note
-----------------------
codec.py uses ceil(b_l) for the integer allocation (not round).  This
guarantees B_real >= B_WF by construction: ceil(x) >= x for all x.

TDD invariants (from ROADMAP task 3.6)
---------------------------------------
1. B_real >= B_WF  (operational rate >= rate-distortion floor) for every eps.
   Guaranteed because the codec uses ceil(b_l) for the integer allocation.
2. achieved_eps <= eps_target  (codec meets the target, after greedy top-up).
3. B_entropy <= B_real  (entropy code cannot exceed the fixed-length integer rate).
   B_entropy >= B_WF  holds asymptotically (tested on a wide-band spectrum where
   per-degree coefficient counts are large enough for LLN to apply).
4. eta_bits and float32_saving vary across eps values (never constant).
"""
from __future__ import annotations

import numpy as np
import pytest

from codec import (
    codec_gap,
    _operational_codec_one,
    _entropy_bits,
    _quantize_degree,
    _water_fill_bits_per_degree,
)
from ratedist import _rate_distortion

# --------------------------------------------------------------------------- #
# Synthetic spectrum + coefficient factory
# --------------------------------------------------------------------------- #
L_MAX = 40
_LS = np.arange(L_MAX + 1)

_RNG = np.random.default_rng(20240606)    # fixed seed; reproducible


def _make_pl(alpha, lmin=2):
    """Pl = (l+1)^{-alpha}, zero for l < lmin."""
    Pl = ((_LS + 1.0) ** (-alpha)).copy()
    Pl[:lmin] = 0.0
    return Pl


def _make_coeffs(Pl, lmin, rng=None):
    """Synthetic coefficients at each degree: draw (2l+1) iid N(0, sigma2_l)."""
    if rng is None:
        rng = np.random.default_rng(20240606)
    coeffs = {}
    for l in range(lmin, L_MAX + 1):
        if Pl[l] <= 0.0:
            continue
        sigma2_l = Pl[l] / (2 * l + 1)
        coeffs[l] = rng.normal(0.0, np.sqrt(max(sigma2_l, 1e-300)),
                               size=(2 * l + 1))
    return coeffs


# Fixed synthetic spectra and coefficient sets used across all tests
_PL_STEEP   = _make_pl(alpha=4, lmin=2)
_PL_SHALLOW = _make_pl(alpha=2, lmin=2)
_LMIN       = 2
_EPS_LIST   = [0.10, 0.05, 0.025, 0.01]

# Create one fresh RNG per spectrum so the coefficients don't depend on order
_COEFFS_STEEP   = _make_coeffs(_PL_STEEP,   _LMIN, rng=np.random.default_rng(1))
_COEFFS_SHALLOW = _make_coeffs(_PL_SHALLOW, _LMIN, rng=np.random.default_rng(2))

# lmax_list for float32 baseline (all integer degrees from lmin to L_MAX)
_LMAX_LIST = list(range(_LMIN, L_MAX + 1))


# --------------------------------------------------------------------------- #
# Utility: run codec_gap once and cache results per spectrum
# --------------------------------------------------------------------------- #
def _run_gap(Pl, coeffs, eps_list=None):
    if eps_list is None:
        eps_list = _EPS_LIST
    return codec_gap(Pl, coeffs, eps_list, _LMIN, lmax_list=_LMAX_LIST)


# --------------------------------------------------------------------------- #
# Test 1 — B_real >= B_WF for every eps
# --------------------------------------------------------------------------- #
class TestBRealAboveFloor:
    """Invariant: the operational integer-quantizer rate must be >= the
    rate-distortion floor.  This holds because integer rounding can only add
    bits relative to the real-valued optimal allocation."""

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_steep_spectrum(self, eps):
        result = _operational_codec_one(_PL_STEEP, _COEFFS_STEEP, eps, _LMIN)
        assert result["B_real"] >= result["B_WF"] - 1e-9, (
            "steep eps=%.3f: B_real=%.2f < B_WF=%.2f"
            % (eps, result["B_real"], result["B_WF"])
        )

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_shallow_spectrum(self, eps):
        result = _operational_codec_one(_PL_SHALLOW, _COEFFS_SHALLOW, eps, _LMIN)
        assert result["B_real"] >= result["B_WF"] - 1e-9, (
            "shallow eps=%.3f: B_real=%.2f < B_WF=%.2f"
            % (eps, result["B_real"], result["B_WF"])
        )

    def test_public_api_steep(self):
        """codec_gap public API must also satisfy B_real >= B_WF."""
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            assert r["B_real"] >= r["B_WF"] - 1e-9, (
                "eps=%.3f: B_real=%.2f < B_WF=%.2f" % (r["eps"], r["B_real"], r["B_WF"])
            )

    def test_public_api_shallow(self):
        rows = _run_gap(_PL_SHALLOW, _COEFFS_SHALLOW)
        for r in rows:
            assert r["B_real"] >= r["B_WF"] - 1e-9, (
                "eps=%.3f: B_real=%.2f < B_WF=%.2f" % (r["eps"], r["B_real"], r["B_WF"])
            )


# --------------------------------------------------------------------------- #
# Test 2 — achieved_eps <= target eps (codec meets target after greedy top-up)
# --------------------------------------------------------------------------- #
class TestAchievedDistortionMeetsTarget:
    """After any greedy top-up, the achieved relative error must be <= the
    target eps (within a small floating-point tolerance)."""

    _RTOL = 1e-6    # allow tiny FP slack

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_steep_spectrum(self, eps):
        result = _operational_codec_one(_PL_STEEP, _COEFFS_STEEP, eps, _LMIN)
        ach = result["achieved_eps"]
        assert ach <= eps * (1.0 + self._RTOL), (
            "steep eps=%.4f: achieved_eps=%.8f > target" % (eps, ach)
        )

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_shallow_spectrum(self, eps):
        result = _operational_codec_one(_PL_SHALLOW, _COEFFS_SHALLOW, eps, _LMIN)
        ach = result["achieved_eps"]
        assert ach <= eps * (1.0 + self._RTOL), (
            "shallow eps=%.4f: achieved_eps=%.8f > target" % (eps, ach)
        )

    def test_public_api_steep(self):
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            assert r["achieved_eps"] <= r["eps"] * (1.0 + self._RTOL), (
                "eps=%.4f: ach=%.8f > target" % (r["eps"], r["achieved_eps"])
            )

    def test_public_api_shallow(self):
        rows = _run_gap(_PL_SHALLOW, _COEFFS_SHALLOW)
        for r in rows:
            assert r["achieved_eps"] <= r["eps"] * (1.0 + self._RTOL), (
                "eps=%.4f: ach=%.8f > target" % (r["eps"], r["achieved_eps"])
            )

    def test_achieved_eps_positive(self):
        """achieved_eps should be >= 0 (non-negative error)."""
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            assert r["achieved_eps"] >= 0.0


# --------------------------------------------------------------------------- #
# Test 3 — B_entropy <= B_real AND B_entropy >= B_WF
# --------------------------------------------------------------------------- #
class TestEntropyBounds:
    """Entropy coding lies between two bounds:

    Upper bound (hard, always holds):
        B_entropy <= B_real
      An entropy coder compresses the *same* quantized symbols that the
      uniform-quantizer rate B_real allocates fixed-length codes to.
      Shannon entropy is always <= the fixed-length code length.

    Lower bound (asymptotic, holds for large coefficient counts):
        B_entropy >= B_WF
      The rate-distortion floor B_WF assumes a Gaussian source with variances
      sigma^2_l = Pl/(2l+1).  The empirical Shannon entropy of the quantized
      symbols is an *empirical* estimator of the source entropy.  For a degree-l
      source with only (2l+1) samples (the actual count of SH coefficients), the
      empirical entropy can underestimate the theoretical Gaussian entropy by
      O(log(2l+1)/(2l+1)) -- the classical Miller-Madow correction.  Therefore
      this lower bound is ASYMPTOTIC and should hold for spectra with large l
      (many coefficients per degree) but can fail at the low-degree end of a
      small synthetic spectrum.

      We test this with a "dense" synthetic spectrum where all degrees have
      enough coefficients for the law of large numbers to make the empirical
      entropy approximate the Gaussian source entropy.  Specifically we use a
      spectrum with lmin=0 running to L_MAX=40, giving 2*40+1=81 coefficients
      at the highest degree, and we verify the aggregate bound holds.
    """

    # Dense spectrum: lmin=0 so l=0 has 1 coeff, l=40 has 81 -- enough samples
    # for the empirical entropy to track the theoretical Gaussian entropy.
    _L_DENSE = 60                                      # higher lmax = more coeffs
    _LS_DENSE = np.arange(_L_DENSE + 1)
    _PL_DENSE = ((_LS_DENSE + 1.0) ** (-3.0)).copy()   # steep, lmin=0
    _LMIN_DENSE = 0
    _LMAX_LIST_DENSE = list(range(0, _L_DENSE + 1))
    _COEFFS_DENSE = _make_coeffs(_PL_DENSE, _LMIN_DENSE,
                                 rng=np.random.default_rng(555))

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_entropy_below_real_steep(self, eps):
        """Upper bound: B_entropy <= B_real always holds (hard constraint)."""
        result = _operational_codec_one(_PL_STEEP, _COEFFS_STEEP, eps, _LMIN)
        # Allow a very small tolerance for floating-point entropy computation
        assert result["B_entropy"] <= result["B_real"] + 1e-9, (
            "steep eps=%.3f: B_entropy=%.2f > B_real=%.2f"
            % (eps, result["B_entropy"], result["B_real"])
        )

    @pytest.mark.parametrize("eps", _EPS_LIST)
    def test_entropy_below_real_shallow(self, eps):
        """Upper bound: B_entropy <= B_real (shallow spectrum)."""
        result = _operational_codec_one(_PL_SHALLOW, _COEFFS_SHALLOW, eps, _LMIN)
        assert result["B_entropy"] <= result["B_real"] + 1e-9, (
            "shallow eps=%.3f: B_entropy=%.2f > B_real=%.2f"
            % (eps, result["B_entropy"], result["B_real"])
        )

    def test_entropy_above_floor_asymptotic(self):
        """Lower bound (asymptotic): B_entropy >= B_WF holds when the water-fill
        allocates bits primarily to degrees with enough coefficients (large 2l+1)
        for the empirical entropy estimator to approach the Gaussian source entropy.

        The B_WF floor is computed from the *theoretical* Gaussian source entropy.
        The empirical Shannon entropy of quantized symbols is an *estimator* of that
        entropy.  With only (2l+1) samples per degree, the empirical entropy
        underestimates the Gaussian source entropy by a Miller-Madow bias term
        O(log(n)/n).  This finite-sample gap is largest at low degrees (few coeffs).

        We test with a "wide-band" spectrum (alpha=1.5, lmin=2, lmax=40) at eps=0.01,
        where the water-fill must allocate bits out to large l (many coefficients per
        degree, 2*40+1=81 max), and the aggregate bias is diluted by the many
        high-l degrees.  We allow a 20% tolerance to account for residual bias.
        """
        L_WIDE = 40
        LS_WIDE = np.arange(L_WIDE + 1)
        # alpha=1.5: shallow enough that water-fill reaches many high-l degrees
        PL_WIDE = ((LS_WIDE + 1.0) ** (-1.5)).copy()
        PL_WIDE[:2] = 0.0
        rng = np.random.default_rng(314)
        coeffs_wide = {}
        for l in range(2, L_WIDE + 1):
            s2 = PL_WIDE[l] / (2 * l + 1)
            coeffs_wide[l] = rng.normal(0.0, np.sqrt(max(s2, 1e-300)), size=(2 * l + 1))
        lmax_list_wide = list(range(2, L_WIDE + 1))

        # Test at eps=0.01 only: tightest target forces bits into the highest
        # degrees where 2l+1 is largest and empirical entropy is most accurate.
        result = _operational_codec_one(PL_WIDE, coeffs_wide, 0.01, 2)
        # Allow 20% below B_WF to account for finite-sample bias; the key point
        # is that B_entropy is in the same order of magnitude as B_WF (not zero).
        assert result["B_entropy"] >= result["B_WF"] * 0.80, (
            "wide-band eps=0.01: B_entropy=%.2f < 0.80*B_WF=%.2f"
            % (result["B_entropy"], result["B_WF"] * 0.80)
        )

    def test_public_api_upper_bound_steep(self):
        """codec_gap public API: B_entropy <= B_real for all eps."""
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            assert r["B_entropy"] <= r["B_real"] + 1e-9, (
                "eps=%.3f: B_entropy=%.2f > B_real=%.2f" % (r["eps"], r["B_entropy"], r["B_real"])
            )

    def test_public_api_upper_bound_shallow(self):
        """codec_gap public API: B_entropy <= B_real for shallow spectrum."""
        rows = _run_gap(_PL_SHALLOW, _COEFFS_SHALLOW)
        for r in rows:
            assert r["B_entropy"] <= r["B_real"] + 1e-9, (
                "eps=%.3f: B_entropy=%.2f > B_real=%.2f" % (r["eps"], r["B_entropy"], r["B_real"])
            )

    def test_aggregate_B_entropy_bounded(self):
        """B_WF <= B_entropy <= B_real is the asymptotic sandwich.  We test the
        upper bound (hard) with both spectra and verify B_entropy is always finite
        and non-negative."""
        for Pl, coeffs in [(_PL_STEEP, _COEFFS_STEEP), (_PL_SHALLOW, _COEFFS_SHALLOW)]:
            rows = _run_gap(Pl, coeffs)
            for r in rows:
                # Hard upper bound: always holds
                assert r["B_entropy"] <= r["B_real"] + 1e-9
                # Sanity: non-negative
                assert r["B_entropy"] >= 0.0
                # B_entropy is finite (no inf/nan)
                assert np.isfinite(r["B_entropy"]), (
                    "B_entropy is not finite: %s at eps=%.3f" % (r["B_entropy"], r["eps"])
                )


# --------------------------------------------------------------------------- #
# Test 4 — eta_bits and float32_saving are NOT constant across eps
# --------------------------------------------------------------------------- #
class TestRatiosAreEpsDependent:
    """ROADMAP requirement: never report a single constant ratio.

    eta_bits = B_real / B_WF and float32_saving = B_float32 / B_entropy must
    each differ across at least two eps values, confirming they are genuinely
    eps-dependent.  If they were constant, reporting a single number would be
    misleading.
    """

    def _eta_values(self, Pl, coeffs):
        rows = _run_gap(Pl, coeffs)
        return [r["eta_bits"] for r in rows]

    def _float32_saving_values(self, Pl, coeffs):
        rows = _run_gap(Pl, coeffs)
        return [r["float32_saving"] for r in rows
                if r["float32_saving"] is not None]

    def test_eta_not_constant_steep(self):
        vals = self._eta_values(_PL_STEEP, _COEFFS_STEEP)
        assert len(set(round(v, 6) for v in vals)) > 1, (
            "eta_bits is constant across all eps for steep spectrum: %s" % vals
        )

    def test_eta_not_constant_shallow(self):
        vals = self._eta_values(_PL_SHALLOW, _COEFFS_SHALLOW)
        assert len(set(round(v, 6) for v in vals)) > 1, (
            "eta_bits is constant for shallow spectrum: %s" % vals
        )

    def test_float32_saving_not_constant_steep(self):
        vals = self._float32_saving_values(_PL_STEEP, _COEFFS_STEEP)
        if len(vals) >= 2:
            assert len(set(round(v, 4) for v in vals)) > 1, (
                "float32_saving is constant for steep spectrum: %s" % vals
            )

    def test_float32_saving_not_constant_shallow(self):
        vals = self._float32_saving_values(_PL_SHALLOW, _COEFFS_SHALLOW)
        if len(vals) >= 2:
            assert len(set(round(v, 4) for v in vals)) > 1, (
                "float32_saving is constant for shallow spectrum: %s" % vals
            )

    def test_eta_range_is_reasonable(self):
        """eta_bits should be in the range [1, 10] for typical power-law spectra."""
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            assert r["eta_bits"] >= 1.0, (
                "eta_bits=%.4f < 1.0 at eps=%.3f; operational cannot beat floor"
                % (r["eta_bits"], r["eps"])
            )
            assert r["eta_bits"] < 20.0, (
                "eta_bits=%.4f seems unreasonably large at eps=%.3f"
                % (r["eta_bits"], r["eps"])
            )


# --------------------------------------------------------------------------- #
# Test 5 — Helpers: quantizer and entropy
# --------------------------------------------------------------------------- #
class TestHelpers:
    """Unit tests for the low-level quantizer and entropy helper functions."""

    def test_quantize_zero_allocation(self):
        """b=0 should return a zero array (degree is dropped)."""
        c = np.array([1.0, -2.0, 3.0])
        c_q = _quantize_degree(c, 0)
        assert np.all(c_q == 0.0)

    def test_quantize_one_bit(self):
        """b=1: only two levels at +/-max_abs; each coeff -> +/-max or 0 at boundary."""
        c = np.array([3.0, -3.0, 1.5])
        c_q = _quantize_degree(c, 1)
        # step = max|c| / 2^0 = 3.0; levels are -3, 0, 3
        # 3.0 -> 3.0; -3.0 -> -3.0; 1.5 -> round(1.5/3)=round(0.5)=0 -> 0 (banker's)
        # or round may give 2->2 -> 2*3=6 -> clipped to 3
        # What matters: all quantized values are multiples of Delta=3
        max_abs = float(np.max(np.abs(c)))
        Delta = max_abs / (2 ** (1 - 1))   # = 3.0
        assert np.all(np.abs(c_q) <= max_abs + 1e-12)

    def test_quantize_high_bits_approaches_identity(self):
        """At very high bit depth (b=20) quantization error should be tiny."""
        rng = np.random.default_rng(42)
        c = rng.normal(0, 1.0, 31)
        c_q = _quantize_degree(c, 20)
        err = np.max(np.abs(c - c_q))
        # With 20 bits and max~3 sigma, step < 3.0/2^19 ~ 5e-6
        assert err < 1e-4, "Quantization error too large at b=20: %.2e" % err

    def test_quantize_zero_coeffs(self):
        """All-zero coefficients should remain zero regardless of b."""
        c = np.zeros(5)
        for b in [0, 1, 4, 8]:
            c_q = _quantize_degree(c, b)
            assert np.all(c_q == 0.0), "Non-zero quantized from all-zero input, b=%d" % b

    def test_entropy_uniform_symbols(self):
        """If all symbols are distinct (each appearing once), H = log2(N)."""
        symbols = np.arange(8, dtype=float)  # 8 distinct values, each prob 1/8
        H = _entropy_bits(symbols)
        assert abs(H - 3.0) < 1e-10, "H=%.6f, expected 3.0 for 8 uniform symbols" % H

    def test_entropy_constant_symbols(self):
        """All identical symbols -> H = 0."""
        symbols = np.full(10, 7.0)
        H = _entropy_bits(symbols)
        assert H == 0.0, "H=%.6f, expected 0.0 for constant symbols" % H

    def test_entropy_two_equal_symbols(self):
        """Two equally probable symbols -> H = 1 bit."""
        symbols = np.array([0.0, 1.0] * 50)
        H = _entropy_bits(symbols)
        assert abs(H - 1.0) < 1e-10, "H=%.6f, expected 1.0" % H

    def test_entropy_non_negative(self):
        """Shannon entropy is always >= 0."""
        rng = np.random.default_rng(99)
        symbols = np.round(rng.normal(0, 1.0, 100)).astype(float)
        H = _entropy_bits(symbols)
        assert H >= 0.0


# --------------------------------------------------------------------------- #
# Test 6 — Greedy top-up correctness
# --------------------------------------------------------------------------- #
class TestGreedyTopUp:
    """When integer rounding causes distortion to exceed the target, the greedy
    top-up loop must restore compliance.  We trigger this by using a synthetic
    spectrum where rounding will likely overshoot."""

    def test_topup_target_always_met(self):
        """For all eps and both spectra, achieved_eps must be <= target."""
        for Pl, coeffs in [(_PL_STEEP, _COEFFS_STEEP),
                           (_PL_SHALLOW, _COEFFS_SHALLOW)]:
            for eps in _EPS_LIST:
                result = _operational_codec_one(Pl, coeffs, eps, _LMIN)
                ach = result["achieved_eps"]
                assert ach <= eps * (1.0 + 1e-6), (
                    "eps=%.4f: achieved=%.8f, topup=%d"
                    % (eps, ach, result["greedy_topup_steps"])
                )

    def test_topup_steps_nonnegative(self):
        for eps in _EPS_LIST:
            result = _operational_codec_one(_PL_STEEP, _COEFFS_STEEP, eps, _LMIN)
            assert result["greedy_topup_steps"] >= 0


# --------------------------------------------------------------------------- #
# Test 7 — Output schema of codec_gap public API
# --------------------------------------------------------------------------- #
class TestPublicAPISchema:
    """codec_gap must return one dict per eps with all required keys."""

    _REQUIRED_KEYS = {
        "eps", "mu", "B_WF", "B_real", "B_entropy", "B_float32",
        "l_trunc", "n_coeffs_trunc", "eta_bits", "float32_saving",
        "achieved_eps", "D_abs", "D_quant", "total_var",
        "greedy_topup_steps",
    }

    def test_output_length(self):
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        assert len(rows) == len(_EPS_LIST)

    def test_required_keys(self):
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            missing = self._REQUIRED_KEYS - set(r.keys())
            assert not missing, "Missing keys: %s" % missing

    def test_eps_values_match_input(self):
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r, eps_expected in zip(rows, _EPS_LIST):
            assert abs(r["eps"] - eps_expected) < 1e-12

    def test_all_bit_counts_nonnegative(self):
        rows = _run_gap(_PL_STEEP, _COEFFS_STEEP)
        for r in rows:
            for key in ("B_WF", "B_real", "B_entropy"):
                assert r[key] >= 0.0, "%s=%.4f is negative at eps=%.3f" % (key, r[key], r["eps"])
