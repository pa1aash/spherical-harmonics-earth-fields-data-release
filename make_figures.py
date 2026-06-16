#!/usr/bin/env python
"""
make_figures.py  --  One-command, end-to-end reproduction of the entire study.

    python make_figures.py

Loads EGM2008 (gravity) and IGRF-13 (magnetic) via pyshtools' built-in datasets,
synthesises the analysed scalar fields (geoid undulation; radial field B_r),
computes the area-weighted relative L2(S2) truncation error eps(l_max) (exact,
deterministic), fits the decay exponent by a weighted least-squares fit of the
degree-variance spectrum (with a delete-one-degree jackknife cross-check),
extracts 10/5/1% threshold degrees, produces Figures 1-5, writes results/ tables,
and runs the Parseval and monotonicity verifications (PASS/FAIL).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

import config as C
import shtrunc as S
import figures as F


def _modver(mod, dist=None):
    """Best-effort version string: __version__ attr, then package metadata,
    then 'unknown'.  Some valid pyshtools builds expose neither, so never raise."""
    v = getattr(mod, "__version__", None)
    if v:
        return v
    try:
        from importlib.metadata import version as _pkgver
        return _pkgver(dist or mod.__name__)
    except Exception:
        return "unknown"


def _versions():
    import scipy, matplotlib, pandas, pyshtools, boule
    v = dict(python=platform.python_version(), numpy=np.__version__,
             scipy=scipy.__version__, matplotlib=matplotlib.__version__,
             pandas=pandas.__version__, pyshtools=_modver(pyshtools),
             boule=_modver(boule))
    try:
        import cartopy
        v["cartopy"] = _modver(cartopy)
    except Exception:
        v["cartopy"] = "unavailable"
    return v


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "End-to-end reproduction of the spherical-harmonic truncation study. "
            "All overrides propagate into config.C before any computation begins."
        )
    )
    p.add_argument(
        "--seed", type=int, default=C.RANDOM_SEED,
        metavar="INT",
        help="Random seed (provenance only; analysis is fully deterministic). "
             "Default: %(default)s",
    )
    p.add_argument(
        "--lref", type=int, default=C.GRAV_LREF,
        metavar="INT",
        help="Gravity reference degree override (also rebuilds GRAV_LMAX_LIST). "
             "Default: %(default)s",
    )
    p.add_argument(
        "--outdir", type=str, default=C.DIR_RESULTS,
        metavar="PATH",
        help="Output directory for results tables. Default: %(default)s",
    )
    p.add_argument(
        "--fields", nargs="+",
        choices=["gravity", "magnetic", "lithospheric"],
        default=["gravity", "magnetic", "lithospheric"],
        metavar="FIELD",
        help="Which fields to analyse. Choices: gravity magnetic lithospheric. "
             "Default: all three.",
    )
    p.add_argument(
        "--no-litho", action="store_true", default=False,
        help="Shorthand to exclude the lithospheric field "
             "(equivalent to omitting 'lithospheric' from --fields).",
    )
    p.add_argument(
        "--save-cache", type=str, default=None,
        metavar="PATH",
        help="If given, save a results cache .pkl file to this path after computing.",
    )
    p.add_argument(
        "--from-cache", type=str, default=None,
        metavar="PATH",
        help="Render ALL figures (1-5 from the .pkl cache, A/B/D from the "
             "results JSON/CSV) with no model download or recomputation, then "
             "exit. Use the cache written by --save-cache (e.g. results/cache.pkl).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------------- from-cache fast path
    # Rebuild every figure from committed artifacts (no download, no recompute):
    # figures 1-5 from the .pkl cache, figures A/B/D from results/*.json|csv.
    if args.from_cache is not None:
        os.makedirs(C.DIR_FIGURES, exist_ok=True)
        print("Rendering all figures from cache %s (no download) ..."
              % args.from_cache)
        F.render_from_cache(args.from_cache, outdir=C.DIR_FIGURES)
        print("Done (figures 1-5 + A/B/D rendered from cache).")
        return 0

    # ---------------------------------------------------------------- apply CLI overrides
    C.RANDOM_SEED = args.seed
    if args.lref != C.GRAV_LREF:
        C.GRAV_LREF = args.lref
        C.GRAV_LMAX_LIST = C._logspaced_degrees(args.lref)

    outdir = args.outdir

    # Resolve effective fields list (--no-litho is a shorthand for dropping lithospheric)
    effective_fields = list(args.fields)
    if args.no_litho and "lithospheric" in effective_fields:
        effective_fields.remove("lithospheric")
    include_litho = "lithospheric" in effective_fields

    os.makedirs(C.DIR_FIGURES, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 72)
    print("Spherical-harmonic spectral truncation of Earth's fields (Seed 1A)")
    print("=" * 72)
    print("RANDOM_SEED = %d   (analysis is fully deterministic; no resampling)"
          % C.RANDOM_SEED)
    versions = _versions()
    print("versions:", ", ".join("%s=%s" % kv for kv in versions.items()))

    # ----------------------------------------------------------------- load
    print("\n[1/5] Loading data and synthesising fields ...")
    grav_field = S.gravity_scalar_field()
    mag_field = S.magnetic_scalar_field()
    fields = [grav_field, mag_field]
    litho_field = S.lithospheric_scalar_field() if (C.INCLUDE_LITHOSPHERIC and include_litho) else None
    if litho_field is not None:
        fields.append(litho_field)
    for f in fields:
        print("   - %-28s l_ref=%-4d grid=%-4d  %s"
              % (f["label"], f["lref"], f["lmax_grid"], "[%s]" % f["units"]))

    # Per-field independent RNGs (analysis is deterministic today; this ensures
    # future stochastic steps cannot introduce field-order dependence).
    root_rng = np.random.default_rng(C.RANDOM_SEED)
    field_rngs = root_rng.spawn(len(fields))  # one independent RNG per field

    # ----------------------------------------------------------------- analyse
    print("\n[2/5] Computing eps(l_max), spectrum/jackknife fits, thresholds ...")
    results = []
    for f in fields:
        r = S.analyse_field(f)
        results.append(r)
        print("   - %-28s p_hat=%+.3f (R2=%.3f)  Parseval=%s  monotonic=%s"
              % (f["label"], r["fit"]["slope"], r["fit"]["r2"],
                 "PASS" if r["parseval"]["passed"] else "FAIL", r["monotonic"]))
    grav_res, mag_res = results[0], results[1]

    # ----------------------------------------------------------------- figures
    print("\n[3/5] Rendering figures ...")
    captions = []
    p, cap = F.figure1_reconstructions(grav_field); captions.append(("Figure 1", cap)); print("   -", os.path.basename(p[0]))
    p, cap = F.figure2_error_curves(results);       captions.append(("Figure 2", cap)); print("   -", os.path.basename(p[0]))
    p, cap = F.figure3_spectra(grav_res, mag_res);  captions.append(("Figure 3", cap)); print("   -", os.path.basename(p[0]))
    p, cap = F.figure4_pareto(results);             captions.append(("Figure 4", cap)); print("   -", os.path.basename(p[0]))
    p, cap = F.figure5_residual(grav_res);          captions.append(("Figure 5", cap)); print("   -", os.path.basename(p[0]))
    # Auxiliary figures A/B/D from the results JSON/CSV (skipped gracefully if a
    # producing script -- codec.py, ratedist.py, radius.py, planetary.py -- has
    # not yet been run). These are copied into the manuscript figures directory.
    captions += F.render_aux_figures()
    cap_path = F.write_captions(captions)
    print("   -", os.path.basename(cap_path))

    # ----------------------------------------------------------------- results
    print("\n[4/5] Writing results tables ...")
    write_results(results, versions, outdir)

    # ----------------------------------------------------------------- cache
    if args.save_cache is not None:
        print("\n[4b/5] Saving results cache to %s ..." % args.save_cache)
        save_results_cache(args.save_cache, results, grav_field)
        print("   - cache written:", args.save_cache)

    # ----------------------------------------------------------------- verify
    print("\n[5/5] Verification")
    ok = verify(results)

    print("\n" + "=" * 72)
    print("HEADLINE NUMBERS")
    print("=" * 72)
    for r in results:
        f = r["field"]
        thr = r["thresholds"]
        thr_s = ", ".join("%s:%s" % ("%g%%" % (t * 100),
                          ("--" if thr[t] is None else "l=%.1f" % thr[t]))
                          for t in C.THRESHOLDS)
        print("%-28s decay exponent p_hat = %+.3f +/- %.3f  (alpha = %.3f +/- %.3f)"
              % (f["label"], r["fit"]["slope"], r["fit"]["slope_se"],
                 r["fit"]["alpha"], r["fit"]["alpha_se"]))
        print("%-28s thresholds: %s" % ("", thr_s))
    print("=" * 72)
    print("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def save_results_cache(path, results, grav_field):
    """Pickle everything figures.py needs to render all 5 figures without
    re-downloading any data.

    The pickle dict contains:
      results_stripped  -- copy of results with each field['s'] set to None
      fig1_maps         -- pre-computed map arrays for figure 1
      fig5_data         -- pre-computed residual map for figure 5
      map_degrees       -- grav_field['map_degrees']
      grav_label        -- grav_field['label']
      grav_lref         -- grav_field['lref']
      grav_map_lmax     -- grav_field['map_lmax']
    """
    import copy
    import pickle

    # ---- results_stripped: deep-copy then null out SHCoeffs objects ----------
    results_stripped = []
    for r in results:
        r2 = copy.deepcopy(r)
        r2["field"]["s"] = None
        results_stripped.append(r2)

    # ---- fig1_maps: one dict per map_degree ---------------------------------
    s_grav = grav_field["s"]
    fig1_maps = []
    for d in grav_field["map_degrees"]:
        lons, lats, data = F.map_arrays(s_grav, d, grav_field["map_lmax"])
        fig1_maps.append({"lmax": d, "lons": lons, "lats": lats, "data": data})

    # ---- fig5_data: residual map at the ~5% threshold -----------------------
    grav_res = results[0]
    l5 = grav_res["thresholds"][0.05]
    l_use = int(round(l5))
    lons_ref, lats_ref, ref_map = F.map_arrays(
        s_grav, grav_field["lref"], grav_field["map_lmax"])
    _, _, trunc_map = F.map_arrays(s_grav, l_use, grav_field["map_lmax"])
    resid = ref_map - trunc_map
    fig5_data = {
        "lons": lons_ref,
        "lats": lats_ref,
        "resid": resid,
        "l_use": l_use,
        "lref": grav_field["lref"],
        "label": grav_field["label"],
    }

    data = {
        "results_stripped": results_stripped,
        "fig1_maps": fig1_maps,
        "fig5_data": fig5_data,
        "map_degrees": grav_field["map_degrees"],
        "grav_label": grav_field["label"],
        "grav_lref": grav_field["lref"],
        "grav_map_lmax": grav_field["map_lmax"],
    }
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


# --------------------------------------------------------------------------- #
def write_results(results, versions, outdir=None):
    import pandas as pd
    if outdir is None:
        outdir = C.DIR_RESULTS
    os.makedirs(outdir, exist_ok=True)

    # thresholds.csv  (marker levels 15/10/5/2.5/1%).  The error curve is the
    # exact area-weighted Parseval ratio, so each crossing degree is a
    # deterministic point estimate -- no interval is attached (the earlier
    # 30%-cell "bootstrap" CI was removed in Phase 1 as statistically invalid).
    rows = []
    for r in results:
        f = r["field"]
        lm = np.asarray(r["curve"]["lmax"])
        cen = S.crossing_degrees(lm, r["curve"]["eps_grid"], C.MARKER_THRESHOLDS)
        for t in C.MARKER_THRESHOLDS:
            l_at = cen[t]
            cr = None if l_at is None else (l_at + 1) ** 2 / (f["lref"] + 1) ** 2
            ncoef = None if l_at is None else (int(np.ceil(l_at)) + 1) ** 2
            rows.append(dict(field=f["key"], field_label=f["label"],
                             threshold=t, threshold_pct=t * 100,
                             lmax=None if l_at is None else round(l_at, 2),
                             frac_coeffs_retained=None if cr is None else round(cr, 6),
                             n_coeffs=ncoef, l_ref=f["lref"]))
    df_thr = pd.DataFrame(rows)
    df_thr.to_csv(os.path.join(outdir, "thresholds.csv"), index=False)

    # epsilon_curves.csv
    crows = []
    for r in results:
        f = r["field"]
        lm = np.asarray(r["curve"]["lmax"])
        cr = S.compression_ratio(lm, f["lref"])
        for i, lv in enumerate(lm):
            crows.append(dict(field=f["key"], lmax=int(lv),
                              eps_grid=r["curve"]["eps_grid"][i],
                              eps_parseval=r["curve"]["eps_parseval"][i],
                              frac_coeffs_retained=cr[i],
                              n_coeffs=int((lv + 1) ** 2)))
    pd.DataFrame(crows).to_csv(
        os.path.join(outdir, "epsilon_curves.csv"), index=False)

    # fit.json
    fit = dict(seed=C.RANDOM_SEED,
               method=("deterministic; PRIMARY decay exponent = directly-measured "
                       "eps-curve log-log slope with a delete-one-degree "
                       "jackknife SE. Companion: weighted-least-squares spectral "
                       "index alpha (sigma_l^2 ~ l^-alpha, weights (2l+1)/2), with "
                       "p=(1-alpha)/2 reported as the asymptotic prediction (valid "
                       "alpha>1; effective index for the band-limited magnetic "
                       "field; inapplicable to the near-white crustal field). "
                       "No bootstrap."),
               versions=versions, fields={})
    for r in results:
        f = r["field"]; ff = r["fit"]
        fit["fields"][f["key"]] = dict(
            label=f["label"], units=f["units"], l_ref=f["lref"],
            decay_exponent=ff["slope"],            # PRIMARY = measured eps-OLS slope
            decay_exponent_jackknife_se=ff["slope_se"],
            decay_exponent_r2=ff["r2"], intercept_log10=ff["intercept"],
            spectrum_alpha=ff["alpha"], spectrum_alpha_se=ff["alpha_se"],
            spectrum_alpha_se_model=ff["alpha_se_model"],
            spectrum_reduced_chi2=ff["spec_chi2"], spectrum_r2=ff["spec_r2"],
            spectrum_p_asymptotic=ff["p_spec"],    # (1-alpha)/2 prediction
            spectrum_p_asymptotic_se=ff["p_spec_se"],
            fit_range=list(f["fit_range"]),
            parseval_max_rel_error=r["parseval"]["max_rel"],
            parseval_pass=bool(r["parseval"]["passed"]),
            monotonic=bool(r["monotonic"]),
            quadrature_area_error=r["qa"]["area_err"],
            quadrature_parseval_rel_error=r["qa"]["rel_err"],
            provenance=f["provenance"])
    with open(os.path.join(outdir, "fit.json"), "w") as fh:
        json.dump(fit, fh, indent=2)

    # summary.txt
    lines = []
    lines.append("Spherical-harmonic spectral truncation of Earth's fields")
    lines.append("Seed 1A: gravity (EGM2008) vs magnetic (IGRF-13)")
    lines.append("=" * 64)
    lines.append("Random seed: %d (provenance only; the analysis is fully "
                 "deterministic -- no resampling)" % C.RANDOM_SEED)
    lines.append("Software: " + ", ".join("%s=%s" % kv for kv in versions.items()))
    lines.append("")
    lines.append("Primary outcome: relative L2(S2) reconstruction error")
    lines.append("  eps(l_max) = ||f - S_{l_max} f|| / ||f||  (area-weighted, GLQ)")
    lines.append("")
    for r in results:
        f = r["field"]; ff = r["fit"]
        lines.append("-" * 64)
        lines.append("FIELD: %s   [analysed scalar field: %s, %s]"
                     % (f["label"], f["field_symbol"], f["units"]))
        lines.append("  provenance: " + f["provenance"])
        lines.append("  reference degree l_ref = %d" % f["lref"])
        lines.append("  decay exponent p_hat = %+.3f +/- %.3f   (PRIMARY: directly-"
                     "measured eps-curve log-log slope over l in [%d,%d],"
                     % (ff["slope"], ff["slope_se"], f["fit_range"][0],
                        f["fit_range"][1]))
        lines.append("      R^2 = %.3f; uncertainty = delete-one-degree jackknife SE)"
                     % ff["r2"])
        lines.append("      spectral index alpha = %+.3f +/- %.3f (weighted fit, "
                     "reduced chi^2 = %.2f, R^2 = %.3f);"
                     % (ff["alpha"], ff["alpha_se"], ff["spec_chi2"], ff["spec_r2"]))
        lines.append("      asymptotic (1-alpha)/2 = %+.3f +/- %.3f -- matches "
                     "p_hat for alpha>1 (gravity); EFFECTIVE index for the"
                     % (ff["p_spec"], ff["p_spec_se"]))
        lines.append("      13-degree magnetic field; NOT applicable to the "
                     "near-white crustal spectrum (alpha < 1).")
        for t in C.THRESHOLDS:
            l_at = r["thresholds"][t]
            if l_at is None:
                lines.append("  eps < %4.0f%%: not reached within l_ref"
                             % (t * 100))
            else:
                lines.append("  eps < %4.0f%%: l_max = %6.1f  (exact crossing),  "
                             "%d coeffs, %.3f%% of reference"
                             % (t * 100, l_at,
                                (int(np.ceil(l_at)) + 1) ** 2,
                                100 * (l_at + 1) ** 2 / (f["lref"] + 1) ** 2))
        lines.append("  VERIFICATION: Parseval %s (max rel err %.2e); "
                     "monotonic %s; quadrature area err %.2e"
                     % ("PASS" if r["parseval"]["passed"] else "FAIL",
                        r["parseval"]["max_rel"],
                        r["monotonic"], r["qa"]["area_err"]))
    lines.append("-" * 64)
    lines.append("NOTES (see AUDIT_REPORT.md):")
    lines.append("  * Fair comparison: each field's eps is measured against ITS "
                 "OWN reference degree (gravity 360, magnetic main field 13). The")
    lines.append("    decay exponent (log-log slope) is the scale-free comparator. "
                 "The IGRF main field decays fastest because it is a deep-core,")
    lines.append("    dipole-dominated signal that is intrinsically band-limited "
                 "to l<=13; the geoid is intermediate; the CRUSTAL magnetic field")
    lines.append("    decays SLOWER than gravity. So 'magnetic compresses better' "
                 "is true ONLY for the core/main field, not magnetism in general.")
    lines.append("  * Gravity threshold sensitivity: the geoid's low-degree power "
                 "(esp. degree 2) depends on the reference-ellipsoid/tide-system")
    lines.append("    treatment; dropping degree 2 raises the 1%% threshold from "
                 "~191 to ~219. The decay exponent and the qualitative gap are")
    lines.append("    robust to this choice. Magnetic results are verified against "
                 "the raw Gauss coefficients to ~1e-13 (no such sensitivity).")
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    _write_numbers_tex(results, outdir)
    print("   - thresholds.csv, epsilon_curves.csv, fit.json, summary.txt, numbers.tex")


# --------------------------------------------------------------------------- #
def _write_numbers_tex(results, outdir=None):
    """Emit results/numbers.tex (and a copy next to main.tex) with \\newcommand
    macros for every headline number.  Rerun make_figures.py + recompile to
    propagate any change end-to-end with zero manual .tex edits."""
    if outdir is None:
        outdir = C.DIR_RESULTS

    def _fmt(v, decimals=3):
        return "%.*f" % (decimals, v)

    def _ncoef_ceil(l_at):
        return (int(np.ceil(l_at)) + 1) ** 2

    def _tex_approx(n):
        """Large coeff count as LaTeX '37{,}249' (comma separator, no ~)."""
        s = "%d" % n
        if len(s) > 3:
            s = s[:-3] + "{,}" + s[-3:]
        return s

    # Pull extra threshold (2.5%) from the stored curve — it is not in the
    # primary C.THRESHOLDS dict but lives in the full epsilon curve.
    def _thr_025(r):
        return S.crossing_degrees(
            r["curve"]["lmax"], r["curve"]["eps_grid"], [0.025])[0.025]

    grav_r, mag_r = results[0], results[1]
    lith_r = results[2] if len(results) > 2 else None

    gfit = grav_r["fit"]; gthr = grav_r["thresholds"]
    mfit = mag_r["fit"];  mthr = mag_r["thresholds"]

    def _field_macros(prefix, fit, thr, r, with_twofive=True):
        """Slope = directly-measured eps-curve log-log slope (PRIMARY); SlopeSE
        its delete-one-degree jackknife SE; Rsq the eps-fit R^2.  Alpha is the
        weighted spectral index (sigma_l^2 ~ l^-alpha); SlopeSpec=(1-alpha)/2 is
        the asymptotic prediction (matches Slope for alpha>1; effective/NA else)."""
        ms = [
            ("\\%sSlope" % prefix,       _fmt(fit["slope"])),
            ("\\%sSlopeSE" % prefix,     _fmt(fit["slope_se"])),
            ("\\%sRsq" % prefix,         _fmt(fit["r2"])),
            ("\\%sAlpha" % prefix,       _fmt(fit["alpha"])),
            ("\\%sAlphaSE" % prefix,     _fmt(fit["alpha_se"])),
            ("\\%sSlopeSpec" % prefix,   _fmt(fit["p_spec"])),
            ("\\%sSlopeSpecSE" % prefix, _fmt(fit["p_spec_se"])),
            ("\\%sLten" % prefix,        _fmt(thr[0.10], 1)),
            ("\\%sLfive" % prefix,       _fmt(thr[0.05], 1)),
        ]
        if with_twofive:
            ms.append(("\\%sLtwofive" % prefix, _fmt(_thr_025(r), 1)))
        ms.append(("\\%sLone" % prefix, _fmt(thr[0.01], 1)))
        return ms

    macros = [("% Gravity (EGM2008 geoid).  Slope = directly-measured eps-curve"
               " log-log slope (PRIMARY); SlopeSE = delete-one-degree jackknife"
               " SE; Alpha = weighted spectral index; SlopeSpec = (1-Alpha)/2"
               " asymptotic prediction.", None)]
    macros += _field_macros("grav", gfit, gthr, grav_r)
    macros.append(("\\gravCoeffsOnePct", _tex_approx(_ncoef_ceil(gthr[0.01]))))
    macros.append(("\\compressionRatio", "%d" % round(_ncoef_ceil(gthr[0.01]) / _ncoef_ceil(mthr[0.01]))))
    macros.append(("% Magnetic main field (IGRF-13 B_r)", None))
    macros += _field_macros("mag", mfit, mthr, mag_r)
    macros.append(("\\magCoeffsOnePct", "%d" % _ncoef_ceil(mthr[0.01])))
    if lith_r:
        lfit = lith_r["fit"]; lthr = lith_r["thresholds"]
        macros.append(("% Crustal magnetic (NGDC-720 B_r)", None))
        macros += _field_macros("lith", lfit, lthr, lith_r, with_twofive=False)
        macros.append(("\\lithCoeffsOnePct", _tex_approx(_ncoef_ceil(lthr[0.01]))))

    # Codec / rate-distortion gap macros (geoid at eps=1%), read straight from
    # results/codec_gap.json so the manuscript never hard-codes a round number.
    #   \codecFloatOverFloor = B_float32 / B_WF   (float32 above the RD floor)
    #   \codecGapGreedy      = B_real    / B_WF   (greedy quantizer above floor)
    #   \codecGapEntropy     = B_entropy / B_WF   (entropy-coded above floor)
    cg_path = os.path.join(outdir, "codec_gap.json")
    if os.path.isfile(cg_path):
        with open(cg_path) as fh:
            cg = json.load(fh)
        g1 = next(r for r in cg["gravity"] if abs(r["eps"] - 0.01) < 1e-9)
        macros.append(("% Codec / rate-distortion gap (geoid at eps=1%), from "
                       "codec_gap.json: float32, greedy quantizer and "
                       "entropy-coded bits relative to the water-fill RD floor.",
                       None))
        macros.append(("\\codecFloatOverFloor", _fmt(g1["B_float32"] / g1["B_WF"], 1)))
        macros.append(("\\codecGapGreedy", _fmt(g1["B_real"] / g1["B_WF"], 2)))
        macros.append(("\\codecGapEntropy", _fmt(g1["B_entropy"] / g1["B_WF"], 2)))

    # Geoid undulation extremes (m) on the SAME grid Figure 1 plots: the gravity
    # field with degrees 0,1 removed, expanded to its reference degree on DH2.
    try:
        gs = grav_r["field"]["s"]
        if gs is not None:
            gdata = gs.pad(grav_r["field"]["map_lmax"]).expand(
                grid="DH2", extend=False).data
            macros.append(("% Geoid undulation extremes (m), DH2 grid at l_ref, "
                           "degrees 0-1 removed (same field Figure 1 plots).",
                           None))
            macros.append(("\\geoidMin", "%d" % round(float(gdata.min()))))
            macros.append(("\\geoidMax", "%d" % round(float(gdata.max()))))
    except Exception:
        pass

    # Dipole power fraction at the analysed epoch (2020.0), B_r degree-variance
    # convention, read straight from temporal_igrf.csv (dipole_power_frac).
    ti_path = os.path.join(outdir, "temporal_igrf.csv")
    if os.path.isfile(ti_path):
        import csv as _csv
        with open(ti_path) as fh:
            frac2020 = next(float(r["dipole_power_frac"])
                            for r in _csv.DictReader(fh) if int(r["year"]) == 2020)
        macros.append(("% Dipole power fraction at epoch 2020.0 (B_r degree "
                       "variance), from temporal_igrf.csv.", None))
        macros.append(("\\dipoleFracTwentyTwenty", _fmt(100.0 * frac2020, 1)))

    # Finite-band check: a pure degree-variance spectrum sigma_l^2 ~ l^-a_nom with
    # a_nom = 1-2p the slope-implied index of the main field, evaluated by exact
    # Parseval and fitted over the SAME l=1..12 window used for the real main
    # field. Demonstrates the 13-degree band is off-asymptotic (slope != -2.92).
    a_nom = round(1.0 - 2.0 * mfit["slope"], 2)
    _l = np.arange(1.0, C.MAG_LREF + 1.0)
    _sig = _l ** (-a_nom)
    _eps = np.array([np.sqrt(_sig[k:].sum() / _sig.sum())
                     for k in range(C.MAG_LREF)])
    _Lk = np.arange(C.MAG_LREF)
    _m = (_Lk >= C.MAG_FIT_RANGE[0]) & (_Lk <= C.MAG_FIT_RANGE[1]) & (_eps > 0)
    _fb = np.polyfit(np.log10(_Lk[_m]), np.log10(_eps[_m]), 1)[0]
    macros.append(("% Finite-band eps-slope of a pure l^-(1-2p) spectrum over the "
                   "main-field fit window l=1..12 (off-asymptotic demonstration).",
                   None))
    macros.append(("\\magFiniteBandSlope", _fmt(_fb, 2)))

    tex_lines = [
        "% numbers.tex -- AUTO-GENERATED by make_figures.py; do not edit by hand.",
        "% Rerun make_figures.py and recompile main.tex to update all headline numbers.",
        "",
    ]
    for cmd, val in macros:
        if val is None:          # comment line
            tex_lines.append("")
            tex_lines.append(cmd)
        else:
            tex_lines.append("\\newcommand{%s}{%s}" % (cmd, val))

    content = "\n".join(tex_lines) + "\n"
    canonical = os.path.join(outdir, "numbers.tex")
    with open(canonical, "w") as fh:
        fh.write(content)
    # Copy next to main.tex so \input{numbers} resolves without a path.
    tex_dir = os.path.join(C.ROOT, "geoid", "geoid_paper_arxiv_overleaf")
    if os.path.isdir(tex_dir):
        with open(os.path.join(tex_dir, "numbers.tex"), "w") as fh:
            fh.write(content)


# --------------------------------------------------------------------------- #
def verify(results):
    ok = True
    # (a) Parseval cross-check
    for r in results:
        f = r["field"]
        status = "PASS" if r["parseval"]["passed"] else "FAIL"
        print("   Parseval  %-26s %s (max rel err %.2e, tol %.0e)"
              % (f["label"], status, r["parseval"]["max_rel"], C.PARSEVAL_RTOL))
        ok &= r["parseval"]["passed"]
    # (b) monotonicity
    for r in results:
        f = r["field"]
        print("   Monotonic %-26s %s"
              % (f["label"], "PASS" if r["monotonic"] else "FAIL"))
        ok &= r["monotonic"]
    # (c) threshold ordering  l@1% > l@5% > l@10%
    for r in results:
        f = r["field"]
        vals = [r["thresholds"][t] for t in C.THRESHOLDS]   # 10,5,1%
        present = [v for v in vals if v is not None]
        ordered = all(x < y for x, y in zip(present, present[1:]))
        print("   Ordering  %-26s %s  (l@10%%<l@5%%<l@1%%: %s)"
              % (f["label"], "PASS" if ordered else "FAIL",
                 " < ".join("%.1f" % v for v in present)))
        ok &= ordered
    # (d) figure files exist
    expected = ["fig1_gravity_reconstructions", "fig2_error_curves",
                "fig3_power_spectra", "fig4_compression_pareto",
                "fig5_gravity_residual"]
    for name in expected:
        for ext in C.FIG_FORMATS:
            path = os.path.join(C.DIR_FIGURES, "%s.%s" % (name, ext))
            exists = os.path.isfile(path) and os.path.getsize(path) > 0
            ok &= exists
            if not exists:
                print("   MISSING FILE:", path)
    print("   Files     all figures present: %s"
          % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(main())
