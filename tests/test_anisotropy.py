"""test_anisotropy.py -- pytest for the best-N-term helpers (task 3.8).

Synthetic 4pi SHCoeffs with a red spectrum, fixed seed, no downloads.
"""
import numpy as np
import pyshtools
import pytest

from shtrunc import truncate_best_N, best_n_term_eps, truncate

L = 30
_power = (np.arange(L + 1) + 1.0) ** (-2.2)
_S = pyshtools.SHCoeffs.from_random(_power, lmax=L, normalization="4pi", seed=7)
_NTOTAL = (L + 1) ** 2                       # number of real coefficients


def _eps_degree(s, lmax):
    """Degree-truncation relative L2 error by Parseval."""
    Pl = s.spectrum()
    return np.sqrt(Pl[lmax + 1:].sum() / Pl.sum())


class TestBestNTermEps:
    def test_keep_all_is_zero(self):
        # Keeping all (L+1)^2 valid coefficients -> eps ~ 0 (float summation-order
        # rounding between cumsum and sum leaves ~1e-8, numerically zero).
        assert best_n_term_eps(_S, [_NTOTAL])[0] == pytest.approx(0.0, abs=1e-6)

    def test_keep_none_is_one(self):
        assert best_n_term_eps(_S, [0])[0] == pytest.approx(1.0, abs=1e-12)

    def test_monotone_non_increasing(self):
        Ns = [1, 5, 20, 100, 300, 600, _NTOTAL]
        e = best_n_term_eps(_S, Ns)
        assert np.all(np.diff(e) <= 1e-12), "best-N-term eps must not increase with N"

    def test_not_worse_than_degree_truncation(self):
        """At matched N=(L'+1)^2, best-N-term eps <= degree-truncation eps (optimality)."""
        for Lp in (3, 8, 15, 25):
            N = (Lp + 1) ** 2
            e_best = best_n_term_eps(_S, [N])[0]
            e_deg = _eps_degree(_S, Lp)
            assert e_best <= e_deg + 1e-12, (
                f"L'={Lp}: best-N-term {e_best:.6e} should be <= degree {e_deg:.6e}")


class TestTruncateBestN:
    def test_keeps_exactly_n_nonzero(self):
        for N in (1, 10, 50, 200):
            out = truncate_best_N(_S, N)
            assert int(np.count_nonzero(out.coeffs)) == N

    def test_keeps_the_largest(self):
        """The kept set must be the N largest-magnitude coefficients."""
        N = 40
        out = truncate_best_N(_S, N)
        p = (_S.coeffs ** 2).ravel()
        thresh = np.sort(p)[::-1][N - 1]          # N-th largest power
        kept = (out.coeffs ** 2).ravel()
        assert np.all(kept[kept > 0] >= thresh - 1e-15)

    def test_eps_matches_best_n_term(self):
        """The eps of the truncate_best_N field equals best_n_term_eps."""
        N = 120
        out = truncate_best_N(_S, N)
        e_field = np.sqrt((_S.coeffs - out.coeffs).__pow__(2).sum() / (_S.coeffs ** 2).sum())
        e_formula = best_n_term_eps(_S, [N])[0]
        assert e_field == pytest.approx(e_formula, rel=1e-10)
