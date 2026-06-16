"""test_shtrunc.py — pytest suite for the core numerics of shtrunc.py.

All three tests operate exclusively on SYNTHETIC spherical-harmonic coefficients
built deterministically from a fixed seed.  No network access, no dataset
downloads, no pyshtools Earth datasets.

Synthetic field
---------------
A real, 4pi-normalised SHCoeffs object with a red power spectrum

    sigma_l^2 ~ (l + 1)^{-2.5},   l = 0 .. L,   L = 40

constructed from a fixed seed (1234).  The field is exactly band-limited to
degree 40 and is guaranteed to exercise the full truncation-error computation.
"""

import numpy as np
import pyshtools
import pytest

from shtrunc import epsilon_curve, expand_on_glq, glq_setup, truncate

# ---------------------------------------------------------------------------
# Shared synthetic field (constructed once for the whole module)
# ---------------------------------------------------------------------------
L = 40
_power = (np.arange(L + 1) + 1.0) ** (-2.5)
_S = pyshtools.SHCoeffs.from_random(
    _power, lmax=L, normalization="4pi", seed=1234
)

# Truncation degrees used in test 2 — four representative points well within
# the band [0, L] so the tail sums are non-trivial.
_LMAX_TEST = [5, 10, 20, 30]


# ---------------------------------------------------------------------------
# Test 1 — GLQ area weights integrate the sphere to 4*pi
# ---------------------------------------------------------------------------
class TestGlqWeightsSumTo4Pi:
    """glq_setup must return area weights that exactly integrate the sphere."""

    @pytest.mark.parametrize("lmax_grid", [20, 40, 90])
    def test_weight_sum(self, lmax_grid):
        """W.sum() equals 4*pi to within 1e-12 for each tested bandwidth."""
        _zeros, _w, W, _nlat, _nlon = glq_setup(lmax_grid)
        assert abs(W.sum() - 4.0 * np.pi) < 1e-12, (
            f"lmax_grid={lmax_grid}: W.sum()={W.sum():.16f} "
            f"differs from 4*pi by {abs(W.sum() - 4*np.pi):.3e}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Independent grid synthesis matches analytic Parseval prediction
# ---------------------------------------------------------------------------
class TestGridEpsMatchesParseval:
    """The area-weighted relative L2 error from explicit grid synthesis must
    equal the analytic Parseval tail-ratio to better than 1e-10.

    This is the key correctness property that justifies the O(l_ref) epsilon
    computation in epsilon_curve: on an exact GLQ grid the two quantities are
    identical up to floating-point rounding.

    The test also verifies that epsilon_curve itself returns values that agree
    with the independently computed grid synthesis, confirming that the
    analytic shortcut in the current implementation is equivalent.
    """

    def setup_method(self):
        """Pre-compute shared grid objects used by all parametric sub-tests."""
        self.zeros, _, W, _nlat, _nlon = glq_setup(L)
        self.Wf = W.ravel()
        self.f_ref = expand_on_glq(_S, self.zeros, L).ravel()
        self.Pl = _S.spectrum()
        # epsilon_curve result for cross-checking
        field = dict(s=_S, lmax_grid=L, lmax_list=_LMAX_TEST)
        self.curve = epsilon_curve(field)

    @pytest.mark.parametrize("lmax", _LMAX_TEST)
    def test_parseval_identity(self, lmax):
        """Grid-synthesised eps matches Parseval tail-ratio to < 1e-10."""
        f_tr = expand_on_glq(truncate(_S, lmax), self.zeros, L).ravel()
        eps_grid = np.sqrt(
            (self.Wf * (self.f_ref - f_tr) ** 2).sum()
            / (self.Wf * self.f_ref ** 2).sum()
        )
        eps_pars = np.sqrt(self.Pl[lmax + 1:].sum() / self.Pl.sum())
        assert abs(eps_grid - eps_pars) < 1e-10, (
            f"lmax={lmax}: eps_grid={eps_grid:.15e}, "
            f"eps_pars={eps_pars:.15e}, diff={abs(eps_grid - eps_pars):.3e}"
        )

    @pytest.mark.parametrize("lmax", _LMAX_TEST)
    def test_epsilon_curve_matches_grid(self, lmax):
        """epsilon_curve's eps_grid entry matches the independent grid value."""
        idx = _LMAX_TEST.index(lmax)
        f_tr = expand_on_glq(truncate(_S, lmax), self.zeros, L).ravel()
        eps_grid_ref = np.sqrt(
            (self.Wf * (self.f_ref - f_tr) ** 2).sum()
            / (self.Wf * self.f_ref ** 2).sum()
        )
        eps_curve = self.curve["eps_grid"][idx]
        assert abs(eps_curve - eps_grid_ref) < 1e-10, (
            f"lmax={lmax}: epsilon_curve eps_grid={eps_curve:.15e}, "
            f"independent grid={eps_grid_ref:.15e}, "
            f"diff={abs(eps_curve - eps_grid_ref):.3e}"
        )


# ---------------------------------------------------------------------------
# Test 3 — eps(l_max) is monotonically non-increasing
# ---------------------------------------------------------------------------
class TestEpsMonotonicNonIncreasing:
    """The truncation-error curve must never increase as lmax grows.

    Adding more degrees to the reconstruction can only reduce or maintain the
    L2 error, so eps must be monotonically non-increasing.  A tolerance of
    1e-12 absorbs floating-point rounding without masking genuine violations.
    """

    def test_monotonic(self):
        """eps_grid is non-increasing across lmax in range(2, L+1)."""
        field = dict(s=_S, lmax_grid=L, lmax_list=list(range(2, L + 1)))
        curve = epsilon_curve(field)
        diffs = np.diff(curve["eps_grid"])
        max_increase = float(diffs.max()) if diffs.size > 0 else 0.0
        assert np.all(diffs <= 1e-12), (
            f"eps_grid is not monotonically non-increasing: "
            f"max positive diff = {max_increase:.3e} at "
            f"lmax={curve['lmax'][int(np.argmax(diffs)) + 1]}"
        )
