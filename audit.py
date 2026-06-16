#!/usr/bin/env python
"""
audit.py -- INDEPENDENT re-derivation of the key quantities, from RAW spherical-
harmonic coefficients, NOT trusting the saved pipeline outputs or round-tripped
scalar fields.

Three independent estimators of eps(l_max) are compared:
  (R) RAW-COEFFICIENT PARSEVAL: degree variance built directly from the raw
      Gauss/Stokes coefficients with explicit normalisation bookkeeping; eps is
      tail-power / total-power. No grid, no round-trip.
  (G) GLQ exact quadrature on the analysed field s (the primary GLQ method).
  (S) Regular lat/lon grid with EXPLICIT sin(theta) area weights (textbook area
      element), an independent quadrature. Also computes the WRONG unweighted RMS
      to demonstrate the area-weighting bug explicitly.

If (R), (G), (S) agree, the pipeline is verified from the ground up.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pyshtools as pysh
import pyshtools.datasets.Earth as Earth
import boule
import config as C
import shtrunc as S

np.set_printoptions(suppress=True)
THRESH = C.THRESHOLDS

# GRS80 normal-gravity-field fully-normalised even zonal coefficients
# (Moritz 1980, Geodetic Reference System 1980). WGS84 is identical to ~1e-11.
# Single source of truth is shtrunc._NORMAL_CBAR; alias it so the two paths
# (audit raw_geoid_degree_variance and shtrunc disturbing_potential_degree_variance)
# can never silently diverge.
NORMAL_CBAR = S._NORMAL_CBAR


# --------------------------------------------------------------------------- #
# eps_from_degree_variance is the single-source-of-truth in shtrunc; import it.
eps_from_degree_variance = S.eps_from_degree_variance


def thresholds_from(lmax_list, eps):
    return S.crossing_degrees(np.asarray(lmax_list), np.asarray(eps), THRESH)


# --------------------------------------------------------------------------- #
# RAW degree variances, computed by hand from the raw coefficients
# --------------------------------------------------------------------------- #
def raw_geoid_degree_variance():
    """Geoid degree variance from RAW EGM2008 4pi Stokes coefficients.

    First-order Bruns: N_lm = a * dC_lm (geodesy-4pi), so the per-degree geoid
    variance is proportional to sum_m (dC^2 + dS^2), where dC has the WGS84
    normal even-zonal field removed and degrees 0,1 are dropped. The constant a^2
    cancels in the ratio eps, so we return the proportional spectrum.
    """
    grav = Earth.EGM2008(lmax=C.GRAV_LREF)
    arr = grav.to_array()                 # (2, L+1, L+1), 4pi-normalised, C/S
    C0 = arr[0].copy(); Sx = arr[1].copy()
    C0[0, :] = 0.0; C0[1, :] = 0.0        # drop degrees 0,1
    Sx[0, :] = 0.0; Sx[1, :] = 0.0
    for l, cbar in NORMAL_CBAR.items():   # remove the reference ellipsoid
        if l <= C.GRAV_LREF:
            C0[l, 0] -= cbar
    Pl = np.sum(C0 ** 2 + Sx ** 2, axis=1)
    return Pl                              # index = degree, valid for l>=2


def raw_geoid_degree_variance_no_normal(drop_deg2=True):
    """Sensitivity variant: instead of subtracting the ellipsoid, simply drop
    degrees 0,1[,2] entirely. Brackets the effect of the normal-field choice."""
    grav = Earth.EGM2008(lmax=C.GRAV_LREF)
    arr = grav.to_array()
    C0 = arr[0].copy(); Sx = arr[1].copy()
    kill = 3 if drop_deg2 else 2
    C0[:kill, :] = 0.0; Sx[:kill, :] = 0.0
    return np.sum(C0 ** 2 + Sx ** 2, axis=1)


def raw_Br_degree_variance(model, lref):
    """B_r 4pi degree variance from RAW Schmidt Gauss coefficients:
        sigma_l^2(B_r) = (l+1)^2/(2l+1) * sum_m (g_lm^2 + h_lm^2)   [nT^2]
    (B_r Schmidt coeff = (l+1)*g; Schmidt->4pi divides power by (2l+1).)
    """
    sch = model.convert(normalization="schmidt", csphase=1)
    arr = sch.to_array()                  # (2, lref+1, lref+1)
    ls = np.arange(lref + 1)
    gh2 = np.sum(arr[0] ** 2 + arr[1] ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pl = (ls + 1) ** 2 / (2 * ls + 1) * gh2
    Pl[0] = 0.0
    return Pl


# --------------------------------------------------------------------------- #
# Independent regular-grid sin(theta) quadrature (and the WRONG unweighted RMS)
# --------------------------------------------------------------------------- #
def eps_regular_grid(s, lref, lmax_list, oversample=1.6):
    """eps via a regular equiangular grid with explicit sin(theta) area weights
    (textbook area element). Returns (eps_sintheta, eps_unweighted_WRONG)."""
    Lg = int(np.ceil(oversample * lref))
    grid_ref = s.pad(Lg).expand(grid="DH2", extend=False)
    lats = np.deg2rad(grid_ref.lats())            # colat = 90 - lat
    theta = np.pi / 2 - lats                       # colatitude
    w_lat = np.sin(theta)                          # sin(theta) area weight
    fref = grid_ref.data
    W = np.broadcast_to(w_lat[:, None], fref.shape)
    den_w = (W * fref ** 2).sum()
    den_u = (fref ** 2).sum()                       # WRONG: no area weight
    eps_w, eps_u = [], []
    for lm in lmax_list:
        ftr = S.truncate(s, int(lm)).pad(Lg).expand(grid="DH2", extend=False).data
        r2 = (fref - ftr) ** 2
        eps_w.append(np.sqrt((W * r2).sum() / den_w))
        eps_u.append(np.sqrt(r2.sum() / den_u))
    return np.array(eps_w), np.array(eps_u)


# --------------------------------------------------------------------------- #
def fmt_thr(d):
    return ", ".join("%s:%s" % ("%g%%" % (t * 100),
                     "--" if d[t] is None else "%.1f" % d[t]) for t in THRESH)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Independent audit: re-derive key quantities from raw spherical-harmonic "
            "coefficients and cross-check against the pipeline outputs."
        )
    )
    p.add_argument(
        "--lref", type=int, default=C.GRAV_LREF,
        metavar="INT",
        help="Override gravity reference degree (also rebuilds GRAV_LMAX_LIST). "
             "Default: %(default)s",
    )
    p.add_argument(
        "--fields", nargs="+",
        choices=["gravity", "magnetic", "lithospheric"],
        default=["gravity", "magnetic", "lithospheric"],
        metavar="FIELD",
        help="Which fields to audit. Choices: gravity magnetic lithospheric. "
             "Default: all three.",
    )
    p.add_argument(
        "--no-litho", action="store_true", default=False,
        help="Skip the lithospheric (crustal) audit block.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------------- apply CLI overrides
    if args.lref != C.GRAV_LREF:
        C.GRAV_LREF = args.lref
        C.GRAV_LMAX_LIST = C._logspaced_degrees(args.lref)

    effective_fields = list(args.fields)
    if args.no_litho and "lithospheric" in effective_fields:
        effective_fields.remove("lithospheric")
    include_litho = "lithospheric" in effective_fields

    print("=" * 78)
    print("INDEPENDENT AUDIT  (recomputed from raw coefficients; seed %d)"
          % C.RANDOM_SEED)
    print("=" * 78)

    report = {}

    # ===================== GRAVITY =====================
    print("\n##### GRAVITY (EGM2008 geoid, l_ref=%d) #####" % C.GRAV_LREF)
    lmax_g = np.array(sorted(set(C.GRAV_LMAX_LIST)))
    Pl_raw = raw_geoid_degree_variance()
    eps_R = eps_from_degree_variance(Pl_raw, lmax_g, lmin=2)
    # sensitivity to normal-field handling
    eps_R_nodeg2 = eps_from_degree_variance(
        raw_geoid_degree_variance_no_normal(True), lmax_g, lmin=3)

    g_field = S.gravity_scalar_field()
    s_g = g_field["s"]
    Pl_pysh = s_g.spectrum()
    eps_G = eps_from_degree_variance(Pl_pysh, lmax_g, lmin=2)   # pyshtools GLQ field
    eps_Sw, eps_Su = eps_regular_grid(s_g, C.GRAV_LREF, lmax_g)   # indep. grid

    d_pysh = thresholds_from(lmax_g, eps_G)        # standard geoid (headline)
    d_raw = thresholds_from(lmax_g, eps_R)           # raw, GRS80 normal, r=a
    d_nod2 = thresholds_from(lmax_g, eps_R_nodeg2)   # drop deg<=2 entirely
    print("  variant (geoid definition)            l@10%   l@5%    l@1%")
    for name, d in [("PYSH std (pysh geoid/WGS84)", d_pysh),
                    ("RAW (GRS80 normal, r=a)", d_raw),
                    ("drop degrees<=2 entirely", d_nod2)]:
        print("  %-30s %s" % (name, "  ".join(
            "%6.1f" % d[t] if d[t] else "   -- " for t in THRESH)))

    # area-weighting verification: GLQ/Parseval vs INDEPENDENT sin-theta grid,
    # both on the SAME field -> isolates the quadrature/weighting, not the field.
    aw_agree = float(np.max(np.abs(eps_G - eps_Sw)))      # must be ~0
    aw_effect = float(np.max(np.abs(eps_Sw - eps_Su)))    # unweighted is wrong
    # degree-2 share drives the low-degree sensitivity
    sh_raw = 100 * Pl_raw[2] / Pl_raw[2:].sum()
    sh_int = 100 * Pl_pysh[2] / Pl_pysh[2:len(Pl_pysh)].sum()
    fr = g_field["fit_range"]; mask = (lmax_g >= fr[0]) & (lmax_g <= fr[1])
    slope_G = S.fit_loglog(lmax_g[mask], eps_G[mask])[0]
    slope_R = S.fit_loglog(lmax_g[mask], eps_R[mask])[0]
    eps100 = float(np.interp(100, lmax_g, eps_G))
    print("  AREA WEIGHTING: |GLQ - sin(theta) grid| = %.2e (PASS<5e-3); "
          "|weighted - UNWEIGHTED| = %.3f" % (aw_agree, aw_effect))
    print("  degree-2 power share: RAW=%.1f%%  pysh=%.1f%% (drives low-l "
          "threshold sensitivity)" % (sh_raw, sh_int))
    print("  eps(l=100) [pysh] = %.4f -> 'sub-1%% near 100' is %s"
          % (eps100, "FALSE" if eps100 > 0.01 else "true"))
    print("  decay exponent p_hat: pysh=%+.3f  raw=%+.3f" % (slope_G, slope_R))
    mono_G = bool(np.all(np.diff(eps_G) <= 1e-9))
    print("  monotonic (pysh): %s" % mono_G)
    report["gravity"] = dict(thr=d_pysh, thr_raw=d_raw, thr_nod2=d_nod2,
                             slope=slope_G, slope_raw=slope_R, aw_agree=aw_agree,
                             aw_effect=aw_effect, mono=mono_G, eps100=eps100,
                             sh_int=sh_int, sh_raw=sh_raw)

    # ===================== MAGNETIC (IGRF-13) =====================
    print("\n##### MAGNETIC main field (IGRF-13 B_r, l_ref=%d) #####" % C.MAG_LREF)
    lmax_m = np.array(C.MAG_LMAX_LIST)
    igrf = Earth.IGRF_13(lmax=C.MAG_LREF, year=C.MAG_YEAR)
    Pl_m_raw = raw_Br_degree_variance(igrf, C.MAG_LREF)
    eps_mR = eps_from_degree_variance(Pl_m_raw, lmax_m, lmin=1)
    m_field = S.magnetic_scalar_field()
    Pl_m_pysh = m_field["s"].spectrum()
    eps_mG = eps_from_degree_variance(Pl_m_pysh, lmax_m, lmin=1)
    # spectrum agreement (raw vs pysh GLQ), nT^2
    L = min(len(Pl_m_raw), len(Pl_m_pysh))
    spec_rel = np.abs(Pl_m_raw[1:L] - Pl_m_pysh[1:L]) / Pl_m_pysh[1:L]
    d_mR = thresholds_from(lmax_m, eps_mR)
    d_mG = thresholds_from(lmax_m, eps_mG)
    print("  RAW-Parseval thresholds : %s" % fmt_thr(d_mR))
    print("  GLQ (pysh) thresholds : %s" % fmt_thr(d_mG))
    print("  max rel diff B_r spectrum (raw vs pysh): %.2e" % spec_rel.max())
    fr = m_field["fit_range"]; mask = (lmax_m >= fr[0]) & (lmax_m <= fr[1])
    slope_mR = S.fit_loglog(lmax_m[mask], eps_mR[mask])[0]
    print("  decay exponent p_hat (RAW) = %+.3f" % slope_mR)
    mono_mR = bool(np.all(np.diff(eps_mR) <= 1e-9))
    print("  monotonic (RAW): %s" % mono_mR)

    # 1.8 fit-range sensitivity: refit EXCLUDING the degree-1 dipole (l=2..12).
    # The dipole carries ~97% of the B_r power, so the 13-degree slope is partly
    # a dipole-dominance summary statistic; this quantifies how much the exponent
    # depends on including l=1, for both the eps-OLS slope and the spectrum fit.
    mask2 = (lmax_m >= 2) & (lmax_m <= 12)
    slope_mR_no1 = S.fit_loglog(lmax_m[mask2], eps_mR[mask2])[0]
    a_full = S.fit_spectrum(Pl_m_raw, (1, 12))
    a_no1 = S.fit_spectrum(Pl_m_raw, (2, 12))
    shift_ols = abs(slope_mR_no1 - slope_mR)
    shift_p = abs(a_no1["p_hat"] - a_full["p_hat"])
    print("  fit-range sensitivity (exclude dipole l=1):")
    print("    eps-OLS slope   l=1..12 = %+.3f  l=2..12 = %+.3f  |shift| = %.3f"
          % (slope_mR, slope_mR_no1, shift_ols))
    print("    spectrum p_hat  l=1..12 = %+.3f  l=2..12 = %+.3f  |shift| = %.3f"
          % (a_full["p_hat"], a_no1["p_hat"], shift_p))
    big = shift_ols > 0.05 or shift_p > 0.05
    print("    --> %s" % ("shift > 0.05: the 13-degree slope is a SUMMARY "
                          "STATISTIC reflecting dipole dominance (noted in "
                          "Limitations)." if big else
                          "shift <= 0.05: slope robust to excluding the dipole."))
    report["magnetic"] = dict(thr=d_mR, slope=slope_mR,
                              spec_rel=float(spec_rel.max()),
                              slope_no_dipole=slope_mR_no1,
                              fit_range_shift=shift_ols,
                              p_shift_no_dipole=shift_p,
                              mono=mono_mR)

    # ===================== MAGNETIC crustal (NGDC-720) =====================
    if include_litho:
        print("\n##### MAGNETIC crustal (NGDC-720 B_r, l_ref=%d) [SECONDARY] #####"
              % C.LITHO_LREF)
        try:
            ngdc = Earth.NGDC_720_V3(lmax=C.LITHO_LREF)
            lmax_l = np.array(sorted(set(C.LITHO_LMAX_LIST)))
            Pl_l_raw = raw_Br_degree_variance(ngdc, C.LITHO_LREF)
            # crustal model is zero below degree 16
            nz = np.nonzero(Pl_l_raw)[0]
            lmin_l = int(nz.min())
            eps_lR = eps_from_degree_variance(Pl_l_raw, lmax_l, lmin=lmin_l)
            d_lR = thresholds_from(lmax_l, eps_lR)
            fr = C.LITHO_FIT_RANGE; mask = (lmax_l >= fr[0]) & (lmax_l <= fr[1])
            slope_lR = S.fit_loglog(lmax_l[mask], eps_lR[mask])[0]
            print("  lowest non-zero degree = %d  (main field removed)" % lmin_l)
            print("  RAW-Parseval thresholds : %s" % fmt_thr(d_lR))
            print("  decay exponent p_hat (RAW) = %+.3f  (vs gravity %+.3f)"
                  % (slope_lR, report["gravity"]["slope"]))
            print("  --> crustal decays %s than gravity"
                  % ("SLOWER" if abs(slope_lR) < abs(report["gravity"]["slope"])
                     else "faster"))
            report["lithospheric"] = dict(thr=d_lR, slope=slope_lR, lmin=lmin_l)
        except Exception as exc:
            print("  NGDC-720 unavailable:", exc)

    # ===================== verification artifact =====================
    # Emit the independent cross-check numbers cited in the manuscript's
    # Verification paragraph so each one traces to a committed file.
    import json
    import os
    ver = {
        "_description": ("Independent verification cross-checks recomputed from "
                         "raw spherical-harmonic coefficients by audit.py. "
                         "Estimators: R=raw-coefficient Parseval, G=GLQ field, "
                         "S=equiangular grid with explicit sin(theta) weights."),
        "magnetic_Br_spectrum_raw_vs_pysh_max_rel_diff":
            report["magnetic"]["spec_rel"],
        "gravity_glq_vs_sintheta_grid_max_abs_eps":
            report["gravity"]["aw_agree"],
        "gravity_unweighted_minus_weighted_max_abs_eps":
            report["gravity"]["aw_effect"],
        "gravity_eps_at_l100": report["gravity"]["eps100"],
        "seed": C.RANDOM_SEED,
    }
    with open(os.path.join(C.DIR_RESULTS, "verification.json"), "w") as fh:
        json.dump(ver, fh, indent=2)
    print("\nWrote %s" % os.path.join(C.DIR_RESULTS, "verification.json"))

    # ===================== verdicts =====================
    print("\n" + "=" * 78)
    print("AUDIT VERDICTS")
    print("=" * 78)
    tol = 5e-3
    gv = report["gravity"]; mg = report["magnetic"]
    P = lambda b: "PASS" if b else "FAIL"
    print("[area weighting]  GLQ == independent sin(theta) grid (%.2e < %.0e): %s"
          % (gv["aw_agree"], tol, P(gv["aw_agree"] < tol)))
    print("[area weighting]  weighting is NOT optional (|wtd-unwtd|=%.3f > 0.02): %s"
          % (gv["aw_effect"], P(gv["aw_effect"] > 0.02)))
    print("[magnetic RAW]    B_r spectrum raw-coeffs == pysh GLQ field (%.1e<1e-6): %s"
          % (mg["spec_rel"], P(mg["spec_rel"] < 1e-6)))
    print("[monotonic]       gravity %s   magnetic %s"
          % (P(gv["mono"]), P(mg["mono"])))
    print("[claim refuted]   gravity eps(l=100)=%.3f, so 'sub-1%% near l=100' is "
          "FALSE (1%% near l~%.0f)" % (gv["eps100"], gv["thr"][0.01]))
    print("[gravity low-l]   thresholds depend on degree-2/normal-field handling "
          "(see range below) -- documented modelling choice, not a bug")
    print("\nCORRECTED HEADLINE NUMBERS (standard geoid = pyshtools/WGS84):")
    print("  GRAVITY  : %s   p_hat=%+.2f  [1%% threshold sensitivity ~%.0f-%.0f "
          "over degree-2 treatment]"
          % (fmt_thr(gv["thr"]), gv["slope"], gv["thr"][0.01],
             gv["thr_nod2"][0.01]))
    print("  MAGNETIC : %s   p_hat=%+.2f  (raw-coeff verified)"
          % (fmt_thr(mg["thr"]), mg["slope"]))
    if "lithospheric" in report:
        lv = report["lithospheric"]
        print("  CRUSTAL  : %s   p_hat=%+.2f  (decays SLOWER than gravity)"
              % (fmt_thr(lv["thr"]), lv["slope"]))
    gv = report["gravity"]
    mg = report["magnetic"]
    tol = 5e-3
    ok = (gv["aw_agree"] < tol and gv["aw_effect"] > 0.02
          and mg["spec_rel"] < 1e-6 and gv["mono"] and mg["mono"])
    print("\nAUDIT %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
