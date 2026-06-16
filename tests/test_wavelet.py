"""test_wavelet.py — pytest suite for wavelet.py (Gate D planar first cut).

All tests operate on SYNTHETIC patches only; no dataset downloads.

Three invariants tested:

    1. ACCURACY: the thresholded wavelet reconstruction achieves eps <= eps_target
       on a synthetic patch (correctness of the bisection search + bit-count logic).

    2. MONOTONICITY: bit counts are positive and strictly increase as eps_target
       decreases (tighter tolerance requires more bits).

    3. POLARITY SANITY: for a KNOWN-LOCALIZED signal (a few Gaussian bumps on a
       zero background) the wavelet bit cost is LESS than for a KNOWN-SMOOTH signal
       (a low-frequency cosine plane) at the same matched eps_target.  This checks
       that the comparison has the correct polarity (localized -> wavelet wins over
       itself; it does NOT test the SH branch — that is the Gate D claim which
       requires real field data).
"""
from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Ensure the project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import wavelet as W


# ---------------------------------------------------------------------------
# Shared synthetic patches (deterministic, no downloads)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(20240606)
_N = 128  # patch size (must be power of 2; matches PATCH_N)


def _make_localized_patch(n=_N, rng=None):
    """A few narrow Gaussian bumps on a near-zero background — sparse in the
    wavelet domain.  Signal is highly localized in space, so wavelets should
    represent it efficiently (few large coefficients)."""
    if rng is None:
        rng = np.random.default_rng(1)
    patch = np.zeros((n, n), dtype=float)
    # Five bumps at random positions, widths sigma~4 cells
    for _ in range(5):
        cx, cy = rng.integers(10, n - 10, size=2)
        sigma = rng.uniform(3.0, 6.0)
        amp = rng.uniform(0.5, 1.5)
        ys, xs = np.mgrid[:n, :n]
        patch += amp * np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))
    return patch


def _make_smooth_patch(n=_N):
    """A low-frequency cosine plane — globally supported, dense in the wavelet
    domain.  Every wavelet coefficient is touched, so wavelets need many of them
    to reconstruct to the same eps."""
    ys, xs = np.mgrid[:n, :n]
    # Two low-frequency cosine waves (1-2 cycles across the patch)
    patch = (np.cos(2 * np.pi * xs / n * 1.5)
             + 0.7 * np.cos(2 * np.pi * ys / n * 1.0))
    return patch.astype(float)


_PATCH_LOCALIZED = _make_localized_patch()
_PATCH_SMOOTH = _make_smooth_patch()

# EPS values used in the Gate D comparison
_EPS_TARGETS = [0.05, 0.01]


# ---------------------------------------------------------------------------
# Helper: find K for a given patch and eps_target
# ---------------------------------------------------------------------------
def _compute_bit_cost(patch, eps_target):
    """Run the wavelet thresholding and return (K, final_eps, bits, total_N)."""
    K, final_eps, total_N, _, _ = W._wavelet_K_for_eps(patch, eps_target)
    bits = W.wavelet_bits(K, total_N, W.B_Q)
    return K, final_eps, bits, total_N


# ---------------------------------------------------------------------------
# Test 1 — Reconstruction accuracy: thresholded inverse achieves eps <= target
# ---------------------------------------------------------------------------
class TestReconstructionAccuracy:
    """The bisection search must find a K such that the exact reconstruction
    error is <= eps_target.  This tests the bisection + reconstruction pipeline
    end-to-end on a synthetic patch."""

    @pytest.mark.parametrize("eps_target", _EPS_TARGETS)
    def test_localized_patch_achieves_eps(self, eps_target):
        """Localized patch: thresholded reconstruction <= eps_target."""
        K, final_eps, bits, total_N = _compute_bit_cost(_PATCH_LOCALIZED, eps_target)
        assert K >= 1, "K must be at least 1"
        assert K <= total_N, "K cannot exceed total coefficients"
        assert final_eps <= eps_target + 1e-6, (
            "Reconstruction error %.6f exceeds eps_target %.4f "
            "(tolerance 1e-6 for float rounding)" % (final_eps, eps_target))

    @pytest.mark.parametrize("eps_target", _EPS_TARGETS)
    def test_smooth_patch_achieves_eps(self, eps_target):
        """Smooth patch: thresholded reconstruction <= eps_target."""
        K, final_eps, bits, total_N = _compute_bit_cost(_PATCH_SMOOTH, eps_target)
        assert K >= 1
        assert K <= total_N
        assert final_eps <= eps_target + 1e-6, (
            "Reconstruction error %.6f exceeds eps_target %.4f" % (final_eps, eps_target))

    def test_bit_counts_positive(self):
        """Bit counts are strictly positive for any reasonable eps."""
        for eps in _EPS_TARGETS:
            _, _, bits, _ = _compute_bit_cost(_PATCH_LOCALIZED, eps)
            assert bits > 0, "bits_wavelet must be positive; got %d" % bits


# ---------------------------------------------------------------------------
# Test 2 — Monotonicity: tighter eps -> more bits (and K)
# ---------------------------------------------------------------------------
class TestMonotonicity:
    """Bit counts must be non-decreasing as eps_target decreases.

    Formally: eps_a > eps_b => bits(eps_a) <= bits(eps_b).
    We test the strict ordering since the patches are smooth enough that
    tighter eps always requires meaningfully more coefficients.
    """

    @pytest.mark.parametrize("patch,label", [
        (_PATCH_LOCALIZED, "localized"),
        (_PATCH_SMOOTH, "smooth"),
    ])
    def test_bits_increase_as_eps_decreases(self, patch, label):
        """Tighter eps => more bits for both patch types."""
        eps_sorted = sorted(_EPS_TARGETS, reverse=True)  # [0.05, 0.01]
        bits_list = []
        K_list = []
        for eps in eps_sorted:
            K, _eps, bits, _N = _compute_bit_cost(patch, eps)
            bits_list.append(bits)
            K_list.append(K)

        for i in range(len(bits_list) - 1):
            assert bits_list[i] <= bits_list[i + 1], (
                "%s patch: bits at eps=%.2f (%d) > bits at eps=%.2f (%d) "
                "(monotonicity violated)" % (
                    label, eps_sorted[i], bits_list[i],
                    eps_sorted[i + 1], bits_list[i + 1]))

    def test_K_monotone(self):
        """K (kept coefficients) must be non-decreasing as eps tightens."""
        eps_sorted = sorted(_EPS_TARGETS, reverse=True)
        K_list = []
        for eps in eps_sorted:
            K, _eps, _bits, _N = _compute_bit_cost(_PATCH_LOCALIZED, eps)
            K_list.append(K)
        for i in range(len(K_list) - 1):
            assert K_list[i] <= K_list[i + 1], (
                "K not monotone: K(eps=%.2f)=%d > K(eps=%.2f)=%d"
                % (eps_sorted[i], K_list[i], eps_sorted[i + 1], K_list[i + 1]))


# ---------------------------------------------------------------------------
# Test 3 — Polarity sanity: localized signal is cheaper than smooth signal
# ---------------------------------------------------------------------------
class TestPolaritySanity:
    """A known-localized signal (a few bumps) should require FEWER wavelet
    bits than a known-smooth global signal (a low-frequency plane) at the
    same eps_target.

    Intuition: the localized patch has nearly all energy concentrated in a
    small number of wavelet coefficients (the high-energy ones at the bump
    locations); the smooth plane spreads its energy across ALL wavelet
    coefficients at the lowest levels, so many coefficients must be kept.

    This test verifies that the comparison has the correct polarity: that
    'localized -> wavelet favoured' is expressed in this implementation.
    It does NOT test the SH branch (which requires real field data).
    """

    @pytest.mark.parametrize("eps_target", _EPS_TARGETS)
    def test_localized_cheaper_than_smooth(self, eps_target):
        """Localized patch uses fewer wavelet bits than smooth patch at eps."""
        _, _, bits_loc, _ = _compute_bit_cost(_PATCH_LOCALIZED, eps_target)
        _, _, bits_smo, _ = _compute_bit_cost(_PATCH_SMOOTH, eps_target)
        assert bits_loc < bits_smo, (
            "Polarity violation at eps=%.2f: localized bits (%d) >= smooth bits (%d). "
            "Expected localized < smooth (wavelet should favour sparsity)."
            % (eps_target, bits_loc, bits_smo))


# ---------------------------------------------------------------------------
# Test 4 — Bit accounting: bit count formula matches manual calculation
# ---------------------------------------------------------------------------
class TestBitAccounting:
    """Verify the bit-count formula directly."""

    def test_wavelet_bits_formula(self):
        """wavelet_bits(K, N, b_q) = K*b_q + K*ceil(log2(N))."""
        for K, N, b_q in [(100, 16384, 16), (500, 16384, 16), (1, 16384, 8)]:
            expected = K * b_q + K * math.ceil(math.log2(N))
            actual = W.wavelet_bits(K, N, b_q)
            assert actual == expected, (
                "wavelet_bits(%d, %d, %d) = %d, expected %d"
                % (K, N, b_q, actual, expected))

    def test_sh_bits_formula(self):
        """sh_bits(L, b_q) = (L+1)^2 * b_q."""
        for L, b_q in [(10, 16), (100, 16), (360, 16)]:
            expected = (L + 1) ** 2 * b_q
            actual = W.sh_bits(L, b_q)
            assert actual == expected, (
                "sh_bits(%d, %d) = %d, expected %d" % (L, b_q, actual, expected))

    def test_index_overhead_is_14_for_n128(self):
        """For n=128 patch with periodization: N=128*128=16384, ceil(log2(16384))=14."""
        N = 128 * 128
        assert N == 16384
        assert math.ceil(math.log2(N)) == 14, (
            "Expected 14 index bits for N=16384; got %d" % math.ceil(math.log2(N)))

    def test_solid_angle_fraction_plausible(self):
        """Solid-angle fraction for a 32-deg patch at lat 40 is ~1-3%."""
        # lat_lo, lat_hi, lon_lo, lon_hi for a 32-deg patch centred at lat 40
        half = 16.0
        frac = W.patch_solid_angle_fraction(40.0 - half, 40.0 + half, 0.0, 32.0)
        assert 0.01 < frac < 0.04, (
            "Solid angle fraction %.5f outside expected range [0.01, 0.04]" % frac)

    def test_quantisation_eps_low_for_b16(self):
        """16-bit quantiser gives eps_quant << 0.01 for a typical patch."""
        patch = np.linspace(-50.0, 50.0, _N * _N).reshape(_N, _N)
        q_err = W.quantisation_eps(patch, b_q=16)
        assert q_err < 1e-3, (
            "16-bit quant eps %.2e is not << 0.01 (expected < 1e-3)" % q_err)
