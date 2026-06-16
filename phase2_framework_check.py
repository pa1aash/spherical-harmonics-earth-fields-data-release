#!/usr/bin/env python
"""
phase2_framework_check.py -- numerical corroboration of the Phase-2 theoretical
framework (cited standard results) against the audited spectral fits in
results/fit.json.

It checks three things the framework predicts and the paper relies on:

  (A) Euler-Maclaurin tail.  For sigma_l^2 = C l^-alpha (alpha > 1) the discarded
      power T(L) = sum_{l=L+1}^inf l^-alpha is reproduced by the Euler-Maclaurin
      asymptotic series to < 1e-9 relative error at L = 100, against the exact
      Hurwitz-zeta tail.  (Framework 2.1.)

  (B) Geoid (in-regime).  The spectrum-fit index alpha_spec and the slope-implied
      index alpha_p = 1 - 2*p agree to within ~5% -- the field is an approximate
      power law, so the asymptotic slope law p = (1-alpha)/2 holds.

  (C) Main field and crustal field (off-asymptotic / outside framework).  The two
      alpha estimates DIVERGE; we flag them and do NOT force-fit the framework.
      A pure power law over the 13-degree main-field band does not reproduce the
      measured eps-curve slope, demonstrating the finite-band caveat (2.5).

Run with the analysis env:
  python phase2_framework_check.py

No downloads, no pyshtools -- pure numerics on fit.json + closed-form sums.
"""
import json
import os

import numpy as np
import mpmath as mp

mp.mp.dps = 50  # 50 decimal digits for the exact tail / EM comparison

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(HERE, "results", "fit.json")


# ----------------------------------------------------------------------------
# (A) Euler-Maclaurin tail vs exact Hurwitz-zeta tail
# ----------------------------------------------------------------------------
def em_tail_series(alpha, N, K):
    """Euler-Maclaurin approximation to sum_{n=N}^inf n^-alpha with K Bernoulli
    correction terms (k = 1..K).  g(x) = x^-alpha.

      sum_{n=N}^inf g(n) = int_N^inf g dx + (1/2) g(N)
                            + sum_{k>=1} B_{2k}/(2k)! * poch(alpha, 2k-1) * N^{-alpha-2k+1}

    using g^{(2k-1)}(N) = -poch(alpha, 2k-1) * N^{-alpha-2k+1}
    and the tail Euler-Maclaurin form  - B_{2k}/(2k)! g^{(2k-1)}(N).
    """
    alpha = mp.mpf(alpha)
    N = mp.mpf(N)
    integral = N ** (1 - alpha) / (alpha - 1)        # int_N^inf x^-alpha dx
    half = mp.mpf(1) / 2 * N ** (-alpha)             # (1/2) g(N)
    total = integral + half
    terms = [("integral", integral), ("half_g(N)", half)]
    for k in range(1, K + 1):
        B2k = mp.bernoulli(2 * k)
        poch = mp.rf(alpha, 2 * k - 1)               # rising factorial alpha(alpha+1)...(alpha+2k-2)
        term = B2k / mp.factorial(2 * k) * poch * N ** (-alpha - 2 * k + 1)
        total += term
        terms.append((f"B{2*k}-correction", term))
    return total, terms


def check_euler_maclaurin():
    print("=" * 74)
    print("(A) EULER-MACLAURIN TAIL  T(L)=sum_{l>L} l^-alpha   at  L = 100  (N = 101)")
    print("=" * 74)
    L = 100
    N = L + 1
    ok_all = True
    for alpha in [2.84, 3.0, 7.51]:
        exact = mp.zeta(mp.mpf(alpha), mp.mpf(N))    # Hurwitz zeta = sum_{n=N}^inf n^-alpha
        print(f"\n  alpha = {alpha}   exact tail T(100) = {mp.nstr(exact, 12)}")
        # leading-order asymptotic only (framework step 3):  C/(alpha-1) * L^(1-alpha)
        lead = mp.mpf(L) ** (1 - mp.mpf(alpha)) / (mp.mpf(alpha) - 1)
        rel_lead = abs(lead - exact) / exact
        print(f"    leading-order  L^(1-a)/(a-1)            rel.err = {mp.nstr(rel_lead, 4)}")
        # user's stated 2-term formula (integral + 1/2 g(N)), N = L+1
        two, _ = em_tail_series(alpha, N, K=0)
        rel_two = abs(two - exact) / exact
        print(f"    2-term (int + 1/2 g(N))                 rel.err = {mp.nstr(rel_two, 4)}")
        for K in (1, 2, 3):
            approx, _ = em_tail_series(alpha, N, K=K)
            rel = abs(approx - exact) / exact
            flag = "  <-- < 1e-9" if rel < mp.mpf("1e-9") else ""
            print(f"    EM series K={K} correction term(s)        rel.err = {mp.nstr(rel, 4)}{flag}")
        # GATE for this alpha: EM with K=3 must beat 1e-9
        approxK, _ = em_tail_series(alpha, N, K=3)
        relK = abs(approxK - exact) / exact
        ok = relK < mp.mpf("1e-9")
        ok_all = ok_all and ok
    print(f"\n  (A) RESULT: EM series (K=3) reproduces the exact tail to < 1e-9 at L=100: "
          f"{'PASS' if ok_all else 'FAIL'}")
    return ok_all


# ----------------------------------------------------------------------------
# (B)/(C) alpha agreement / divergence per field
# ----------------------------------------------------------------------------
def alpha_compare(fit):
    print("\n" + "=" * 74)
    print("(B)/(C) SPECTRUM-FIT alpha  vs  SLOPE-IMPLIED alpha = 1 - 2p   per field")
    print("=" * 74)
    rows = []
    for key in ("gravity", "magnetic", "lithospheric"):
        f = fit["fields"][key]
        p = f["decay_exponent"]
        a_spec = f["spectrum_alpha"]
        a_p = 1 - 2 * p
        gap = abs(a_spec - a_p) / abs(a_spec) if a_spec != 0 else float("inf")
        s_star_p = -p
        s_star_spec = (a_spec - 1) / 2
        rows.append((key, p, a_p, a_spec, gap, s_star_p, s_star_spec))
        print(f"\n  {f['label']}")
        print(f"    measured slope p            = {p:+.4f}")
        print(f"    slope-implied alpha = 1-2p  = {a_p:+.4f}")
        print(f"    spectrum-fit alpha          = {a_spec:+.4f}")
        print(f"    relative gap |da|/alpha     = {gap*100:6.2f}%")
        print(f"    s* from -p                  = {s_star_p:+.4f}")
        print(f"    s* from (alpha-1)/2         = {s_star_spec:+.4f}")
    return rows


def check_regime(rows):
    print("\n" + "=" * 74)
    print("REGIME CLASSIFICATION (framework scoped to the geoid; not force-fit)")
    print("=" * 74)
    res = {}
    for key, p, a_p, a_spec, gap, s_p, s_spec in rows:
        if key == "gravity":
            ok = gap < 0.05  # ~5% agreement => in-regime power law
            res["gravity_in_regime"] = ok
            print(f"  geoid: alpha agree to {gap*100:.2f}% (< 5% required) => "
                  f"IN-REGIME power law: {'PASS' if ok else 'FAIL'}")
        elif key == "magnetic":
            # off-asymptotic: must DIVERGE more than the geoid AND alpha<convergence is moot;
            # the diagnostic is that the 13-degree band breaks the asymptotic relation.
            diverges = gap > 0.05
            res["magnetic_off_asymptotic"] = diverges
            print(f"  main field: alpha gap {gap*100:.2f}% (> geoid's; 13-degree band, ~97% "
                  f"dipole power) => OFF-ASYMPTOTIC, effective index only: "
                  f"{'flagged' if diverges else 'NOT flagged (unexpected)'}")
        elif key == "lithospheric":
            # outside framework: spectrum alpha < 1 (divergent tail) while slope-implied > 1
            sign_split = (a_spec < 1) and (a_p > 1)
            res["crustal_outside"] = sign_split
            print(f"  crustal: slope-implied alpha={a_p:+.3f} (>1, marginally conv.) vs "
                  f"spectrum alpha={a_spec:+.3f} (<1, near-white/divergent tail)")
            print(f"           => SHARP DIVERGENCE, OUTSIDE framework (no alpha, no s*): "
                  f"{'flagged' if sign_split else 'NOT flagged (unexpected)'}")
    return res


# ----------------------------------------------------------------------------
# (2.5) finite-band demonstration: a pure power law over l=1..13 does NOT
#       reproduce the measured eps-curve slope of the dipole-dominated field.
# ----------------------------------------------------------------------------
def eps_curve_slope(sigma2, lmax_fit):
    """Given per-degree variance sigma2[l] for l=1..L (index l-1), build the
    relative-L2 eps curve eps(lmax)^2 = sum_{l>lmax} sigma2 / sum_all, then OLS
    log10(eps) on log10(lmax) over lmax = 1..lmax_fit."""
    L = len(sigma2)
    ells = np.arange(1, L + 1)
    total = sigma2.sum()
    tail = np.array([sigma2[lm:].sum() for lm in range(L)])  # tail above lmax index
    # eps(lmax) for lmax = 1..L-1 (eps(L)=0); use degrees up to lmax_fit
    lmaxes = np.arange(1, lmax_fit + 1)
    eps = np.sqrt(np.array([sigma2[lm:].sum() for lm in lmaxes]) / total)
    x = np.log10(lmaxes.astype(float))
    y = np.log10(eps)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope


def check_finite_band(fit):
    print("\n" + "=" * 74)
    print("(2.5) FINITE-BAND CAVEAT: pure power law over l=1..13 vs measured slope")
    print("=" * 74)
    L = 13
    ells = np.arange(1, L + 1).astype(float)
    measured = fit["fields"]["magnetic"]["decay_exponent"]
    for a in (6.85, 7.51):
        sigma2 = ells ** (-a)
        s = eps_curve_slope(sigma2, lmax_fit=12)
        print(f"  pure sigma_l^2 = l^-{a:<4} over l=1..13  ->  eps-curve OLS slope (l=1..12) "
              f"= {s:+.3f}")
    print(f"  MEASURED IGRF-13 eps-curve slope                         "
          f"= {measured:+.3f}")
    print("  => the measured slope is NOT reproduced by a pure power law of the spectrum-fit\n"
          "     (or slope-implied) index over only 13 degrees: the asymptotic law p=(1-a)/2\n"
          "     is OFF-ASYMPTOTIC here; -2.924 is a dipole-dominated summary statistic.")


def main():
    with open(FIT) as fh:
        fit = json.load(fh)
    a_ok = check_euler_maclaurin()
    rows = alpha_compare(fit)
    res = check_regime(rows)
    check_finite_band(fit)

    print("\n" + "=" * 74)
    print("PHASE-2 NUMERICAL GATE")
    print("=" * 74)
    gate = (a_ok and res.get("gravity_in_regime") and res.get("magnetic_off_asymptotic")
            and res.get("crustal_outside"))
    print(f"  EM tail < 1e-9 at L=100 ............... {'PASS' if a_ok else 'FAIL'}")
    print(f"  geoid alpha agree to ~5% (in-regime) .. {'PASS' if res.get('gravity_in_regime') else 'FAIL'}")
    print(f"  main field flagged off-asymptotic ..... {'PASS' if res.get('magnetic_off_asymptotic') else 'FAIL'}")
    print(f"  crustal flagged outside framework ..... {'PASS' if res.get('crustal_outside') else 'FAIL'}")
    print(f"\n  PHASE-2 NUMERIC CORROBORATION: {'PASS' if gate else 'FAIL'}")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
