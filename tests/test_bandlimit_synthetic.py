"""test_bandlimit_synthetic.py — Synthetic unit tests for bandlimit.py helpers.

All tests use only numpy.  No downloads, no pyshtools Earth datasets.

Synthetic construction
----------------------
For a power law signal  S_l = l^{-3}  and constant noise  N_l = const = C:

    Per-degree crossover (SNR_l = 1):
        l^{-3} = C  =>  l* = C^{-1/3}

    Cumulative-tail crossover:
        sum_{k > l} k^{-3}  <  sum_{k > l} C
        (L - l) * C  >  sum_{k=l+1}^{L} k^{-3}

The tests set up a discrete version on degrees 1..L and assert that
_perdegree_crossover and _cumulative_tail_crossover return the expected
degree (analytically or numerically derivable without model downloads).
"""

import numpy as np
import pytest
import sys
import os

# Make the project root importable so we can import bandlimit directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bandlimit import (
    _perdegree_crossover,
    _cumulative_tail_crossover,
    _first_below_snr,
)


# ---------------------------------------------------------------------------
# Shared synthetic arrays
# ---------------------------------------------------------------------------
L = 100                         # maximum degree (large enough to be conclusive)
ls = np.arange(L + 1, dtype=float)
ls[0] = 1.0                     # avoid l=0 division in power law (not used below)

# Signal: S_l = l^{-3}  for l = 1..L, S_0 = 0
S_power = np.zeros(L + 1)
S_power[1:] = ls[1:] ** (-3.0)

# Noise: constant C chosen so crossover occurs at a known interior degree.
# C = S_l* = l*^{-3}  =>  l* = C^{-1/3}.
# Choose l* = 20: C = 20^{-3} = 1/8000.
L_STAR = 20
C_NOISE = L_STAR ** (-3.0)       # = 1/8000

# Constant-noise array
N_const = np.full(L + 1, C_NOISE)
N_const[0] = 0.0                 # degree 0 not used

# Verify the analytic expectation: at l=L_STAR, S[L_STAR] == N[L_STAR] exactly
# (by construction), so the first l where S < N is L_STAR+1 (if S_l is strictly
# decreasing beyond L_STAR, which it is for l^{-3}).
# At l=L_STAR: S[L_STAR] = L_STAR^{-3} = C_NOISE = N_const[L_STAR] -> equal.
# At l=L_STAR+1: S[L_STAR+1] = (L_STAR+1)^{-3} < L_STAR^{-3} = C_NOISE -> S < N.
# => expected per-degree crossover = L_STAR + 1 = 21.
EXPECTED_PERDEGREE = L_STAR + 1  # = 21


# ---------------------------------------------------------------------------
# Test 1 — Per-degree crossover on the power-law vs constant-noise synthetic
# ---------------------------------------------------------------------------
class TestPerdegCrossover:
    """_perdegree_crossover must return the analytically expected degree."""

    def test_known_crossover(self):
        """Per-degree crossover = L_STAR+1 for S_l = l^{-3}, N_l = L_STAR^{-3}."""
        result = _perdegree_crossover(S_power, N_const, l_min=1)
        assert result == EXPECTED_PERDEGREE, (
            "Expected per-degree crossover at l=%d; got %r"
            % (EXPECTED_PERDEGREE, result)
        )

    def test_no_crossover_when_signal_always_above(self):
        """Returns None when signal exceeds noise everywhere in [l_min, L]."""
        # Noise far below signal everywhere
        N_tiny = np.full(L + 1, 1e-20)
        result = _perdegree_crossover(S_power, N_tiny, l_min=1)
        assert result is None, (
            "Expected None (no crossover); got %r" % result
        )

    def test_immediate_crossover_at_l_min(self):
        """Returns l_min when noise exceeds signal from the first considered degree."""
        # Noise far above signal everywhere
        N_huge = np.full(L + 1, 1e10)
        result = _perdegree_crossover(S_power, N_huge, l_min=1)
        assert result == 1, (
            "Expected crossover at l=1 (noise dominates from start); got %r" % result
        )

    def test_crossover_respects_l_min(self):
        """l_min is honoured: degrees below l_min are not checked even if S < N there."""
        # Noise large only at degrees 1..10 (below l_min=15), and very small above
        N_selective = np.zeros(L + 1)
        N_selective[1:11] = 1e10       # dominates for l=1..10
        N_selective[11:] = 1e-30       # negligible above
        # With l_min=15, should return None (no crossover above l_min)
        result = _perdegree_crossover(S_power, N_selective, l_min=15)
        assert result is None, (
            "Expected None when noise only dominates below l_min; got %r" % result
        )


# ---------------------------------------------------------------------------
# Test 2 — Cumulative-tail crossover
# ---------------------------------------------------------------------------
class TestCumulativeTailCrossover:
    """_cumulative_tail_crossover must find the correct degree numerically."""

    def test_cumtail_crossover_is_nonnone_for_dominated_tail(self):
        """Cumulative tail crossover is well-defined when the noise tail
        eventually exceeds the signal tail.

        For S_l = l^{-3} and N_l = constant C over degrees 1..L, the constant
        noise accumulates faster in the tail than the decaying signal, so the
        cumulative tail crossover can occur BEFORE (at a smaller degree than)
        the per-degree crossover.  The per-degree crossover is l=21 in this
        synthetic; the cumulative crossover is l=6 (verified numerically).

        This test asserts that:
          (a) both crossovers are found (non-None),
          (b) the cumulative crossover is <= per-degree crossover,
              because the tail aggregates noise from ALL future degrees, so the
              total noise tail can dominate even when individual SNR_l > 1.
          (c) the cumulative crossover is at the analytically expected degree 6.
        """
        l_pd = _perdegree_crossover(S_power, N_const, l_min=1)
        l_ct = _cumulative_tail_crossover(S_power, N_const, l_min=1)
        # Both should be non-None for this synthetic
        assert l_pd is not None, "Per-degree crossover unexpectedly None"
        assert l_ct is not None, "Cumulative crossover unexpectedly None"
        # Cumulative crossover is the EARLIER (stricter) condition here:
        # the noise tail exceeds the signal tail before individual SNR drops to 1.
        assert l_ct <= l_pd, (
            "Cumulative crossover (%d) should be <= per-degree crossover (%d) "
            "for constant noise with decaying signal (more noise in future tail)"
            % (l_ct, l_pd)
        )
        # Verify the numerically expected value
        assert l_ct == 6, (
            "Expected cumulative crossover at l=6 for S_l=l^-3, "
            "N_l=20^-3; got %r" % l_ct
        )

    def test_no_cumtail_crossover_when_signal_always_dominant(self):
        """Returns None when cumulative signal tail > cumulative noise tail always."""
        # Noise so tiny the signal tail is always larger
        N_tiny = np.full(L + 1, 1e-20)
        result = _cumulative_tail_crossover(S_power, N_tiny, l_min=1)
        assert result is None, (
            "Expected None (signal dominates cumulative); got %r" % result
        )

    def test_cumtail_manual(self):
        """Manually verify the cumulative tail crossover on a tiny 5-degree system.

        System: degrees 0..4, l_min=1.
        S = [0, 4, 2, 1, 0.5]    (padded with 0 at degree 0)
        N = [0, 0.1, 0.1, 0.1, 3]
        tail_S: tail_S[l] = sum_{k>l} S[k]
            tail_S[0]=7.5, tail_S[1]=3.5, tail_S[2]=1.5, tail_S[3]=0.5, tail_S[4]=0
        tail_N:
            tail_N[0]=3.3, tail_N[1]=3.2, tail_N[2]=3.1, tail_N[3]=3.0, tail_N[4]=0
        Crossover: first l where tail_S[l] < tail_N[l]:
            l=1: 3.5 > 3.2 -> no
            l=2: 1.5 < 3.1 -> YES  =>  expected = 2
        """
        S_small = np.array([0.0, 4.0, 2.0, 1.0, 0.5])
        N_small = np.array([0.0, 0.1, 0.1, 0.1, 3.0])
        result = _cumulative_tail_crossover(S_small, N_small, l_min=1)
        assert result == 2, (
            "Expected cumulative crossover at l=2; got %r" % result
        )


# ---------------------------------------------------------------------------
# Test 3 — _first_below_snr helper
# ---------------------------------------------------------------------------
class TestFirstBelowSnr:
    """_first_below_snr must find the first degree where SNR_l < threshold."""

    def test_known_snr_threshold(self):
        """SNR_l = l^{-3} / C_NOISE = l^{-3} * L_STAR^3.
        SNR drops below 10 when l^{-3} * L_STAR^3 < 10
          => l > L_STAR / 10^{1/3} ~ 20/2.154 ~ 9.28  => first integer >= 10.
        """
        # SNR_l = S_l / N_l = l^{-3} / C_NOISE = l^{-3} * L_STAR^3
        # SNR < 10 when l^3 > L_STAR^3 / 10 => l > L_STAR * 10^{-1/3}
        threshold_10 = 10.0
        # L_STAR*(10^{-1/3}) = 20/2.1544... = 9.283..., so the first integer l
        # with SNR<10 is l=10:
        #   SNR(10) = 10^{-3} * 20^3 = 8000/1000 = 8   < 10  YES
        #   SNR(9)  =  9^{-3} * 20^3 = 8000/729  = 10.97 > 10  NO
        result = _first_below_snr(S_power, N_const, threshold=threshold_10, l_min=1)
        assert result == 10, (
            "Expected SNR<10 first at l=10; got %r" % result
        )

    def test_snr_threshold_1_matches_perdegree(self):
        """_first_below_snr at threshold=1 should match _perdegree_crossover."""
        result_snr = _first_below_snr(S_power, N_const, threshold=1.0, l_min=1)
        result_pd = _perdegree_crossover(S_power, N_const, l_min=1)
        assert result_snr == result_pd, (
            "_first_below_snr(threshold=1) = %r, _perdegree_crossover = %r"
            % (result_snr, result_pd)
        )

    def test_returns_none_when_snr_never_drops(self):
        """Returns None when SNR stays above the threshold for all degrees."""
        N_tiny = np.full(L + 1, 1e-30)
        N_tiny[0] = 0.0
        # SNR = S_l / 1e-30 is astronomically large; never drops below 10
        result = _first_below_snr(S_power, N_tiny, threshold=10.0, l_min=1)
        assert result is None, (
            "Expected None (SNR never drops below 10); got %r" % result
        )
