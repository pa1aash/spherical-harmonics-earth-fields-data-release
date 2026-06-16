"""
figures.py  --  All five figures for the spectral-truncation study.

Each figure is saved as a 300-dpi PNG and a vector PDF in figures/, and each
figure function returns a ready-to-paste caption string.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as C
import shtrunc as S

# --------------------------------------------------------------------------- #
# Global, readable, colourblind-friendly style
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": C.DPI,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})

# cartopy is optional at plot time; degrade gracefully if coastlines fail.
try:
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    _HAVE_CARTOPY = True
except Exception:                                       # pragma: no cover
    _HAVE_CARTOPY = False


def _save(fig, name):
    paths = []
    for ext in C.FIG_FORMATS:
        p = os.path.join(C.DIR_FIGURES, "%s.%s" % (name, ext))
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Map helpers
# --------------------------------------------------------------------------- #
def map_arrays(s, lmax, map_lmax):
    """Return (lons[-180..180], lats, data) for s truncated at lmax, on a DH2
    regular grid of bandwidth map_lmax."""
    g = S.truncate(s, lmax).pad(map_lmax).expand(grid="DH2", extend=False)
    data, lons, lats = g.data, g.lons(), g.lats()
    shift = (((lons + 180) % 360) - 180)
    order = np.argsort(shift)
    return shift[order], lats, data[:, order]


def _draw_map(ax, lons, lats, data, vmax, cmap, title=None):
    # rasterized=True keeps the (large) colour mesh as a 300-dpi raster inside
    # the otherwise-vector PDF -> crisp text/coastlines, small file size.
    if _HAVE_CARTOPY:
        d, lo = add_cyclic_point(data, coord=lons)
        mesh = ax.pcolormesh(lo, lats, d, cmap=cmap, vmin=-vmax, vmax=vmax,
                             shading="auto", transform=ccrs.PlateCarree(),
                             rasterized=True)
        try:
            ax.coastlines(linewidth=0.45, color="#222222")
        except Exception:
            pass
        ax.set_global()
        gl = ax.gridlines(draw_labels=False, linewidth=0.4, color="gray",
                          alpha=0.4)
    else:                                               # pragma: no cover
        mesh = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=-vmax, vmax=vmax,
                             shading="auto", rasterized=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    if title:
        ax.set_title(title)
    return mesh


def _geoaxes(fig, spec):
    if _HAVE_CARTOPY:
        return fig.add_subplot(spec, projection=ccrs.PlateCarree())
    return fig.add_subplot(spec)


# --------------------------------------------------------------------------- #
# Figure 1 -- gravity geoid reconstructed at four truncation degrees
# --------------------------------------------------------------------------- #
def figure1_reconstructions(grav, precomputed_maps=None):
    """Render figure 1.

    Parameters
    ----------
    grav : dict
        Gravity field dict (must contain 's', 'map_degrees', 'map_lmax').
    precomputed_maps : list of dict, optional
        If provided, each dict must have keys 'lmax', 'lons', 'lats', 'data'.
        When supplied these arrays are used directly instead of calling
        map_arrays() on grav['s'].
    """
    degs = grav["map_degrees"]
    if precomputed_maps is not None:
        maps = [(m["lons"], m["lats"], m["data"]) for m in precomputed_maps]
    else:
        s = grav["s"]
        maps = [map_arrays(s, d, grav["map_lmax"]) for d in degs]
    full = maps[-1][2]
    vmax = float(np.ceil(np.percentile(np.abs(full), 99.5) / 10) * 10)

    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.06)
    mesh = None
    for k, (d, (lons, lats, data)) in enumerate(zip(degs, maps)):
        ax = _geoaxes(fig, gs[k // 2, k % 2])
        mesh = _draw_map(ax, lons, lats, data, vmax, C.DIVERGING_CMAP,
                         title=r"$\ell_{\max}=%d$" % d)
    cbar = fig.colorbar(mesh, ax=fig.axes, orientation="horizontal",
                        fraction=0.05, pad=0.04, shrink=0.7, extend="both")
    cbar.set_label("Geoid undulation $N$ (m)  [WGS84 ellipsoid removed]")
    fig.suptitle("EGM2008 geoid reconstructed at increasing truncation degree",
                 y=0.98, fontsize=14)
    paths = _save(fig, "fig1_gravity_reconstructions")
    cap = ("Figure 1. Earth's geoid undulation $N$ (height of the geoid above "
           "the WGS84 ellipsoid, metres) synthesised from EGM2008 at four "
           "spherical-harmonic truncation degrees $\\ell_{\\max}\\in"
           "\\{%s\\}$. All panels share the diverging colour scale "
           "($\\pm%.0f$ m, centred at zero); degrees 0--1 are removed. Coarse "
           "truncations capture only continental-scale undulations, while "
           "successive degrees add progressively finer structure."
           % (", ".join(str(d) for d in degs), vmax))
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure 2 -- epsilon(l_max) for both fields, log-log, CI bands, fits, thresholds
# --------------------------------------------------------------------------- #
def figure2_error_curves(results):
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for r in results:
        f = r["field"]
        sec = f.get("secondary", False)
        lm = np.asarray(r["curve"]["lmax"], float)
        eg = r["curve"]["eps_grid"]
        pos = eg > 0
        # eps(l_max) is the exact area-weighted Parseval ratio (deterministic, no
        # sampling); the headline uncertainty is on the decay exponent, shown as
        # p_hat +/- SE (from the weighted spectrum fit).
        lab = "%s ($\\hat{p}=%.2f\\pm%.2f$)%s" % (
            f["label"], r["fit"]["slope"], r["fit"]["slope_se"],
            "  [secondary]" if sec else "")
        if sec:
            ax.plot(lm[pos], eg[pos], "--", color=f["color"], lw=1.4,
                    alpha=0.9, label=lab)
        else:
            ax.plot(lm[pos], eg[pos], "o-", color=f["color"], ms=4, lw=1.8,
                    label=lab)
            # fitted eps power law over the fit range (slope=intercept matched)
            flo, fhi = f["fit_range"]
            xs = np.linspace(max(flo, lm[pos].min()), fhi, 50)
            ys = 10 ** r["fit"]["intercept"] * xs ** r["fit"]["slope"]
            ax.plot(xs, ys, "--", color=C.COLOR_FIT, lw=1.0, alpha=0.8)
            # threshold crossing markers
            for t in C.THRESHOLDS:
                l_at = r["thresholds"][t]
                if l_at is not None:
                    ax.plot([l_at], [t], marker="s", color=f["color"], ms=7,
                            mec="k", mew=0.6, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    # threshold guides + labels at the right edge
    for t in C.THRESHOLDS:
        ax.axhline(t, color=C.COLOR_THRESH, lw=0.8, ls=":", alpha=0.7)
        ax.text(0.995, t, " %g%%" % (t * 100), color=C.COLOR_THRESH,
                fontsize=9, va="bottom", ha="right",
                transform=ax.get_yaxis_transform())

    ax.set_xlabel(r"Truncation degree  $\ell_{\max}$")
    ax.set_ylabel(r"Relative $L^2(S^2)$ reconstruction error  $\varepsilon$")
    ax.set_title("Spectral-truncation error: gravity vs magnetic field")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.text(0.985, 0.97,
            "curve = exact area-weighted error\n"
            "black dashed = log-log $\\varepsilon$-OLS fit\n"
            "squares = 10 / 5 / 1% threshold",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    paths = _save(fig, "fig2_error_curves")
    cap = ("Figure 2. Relative $L^2(S^2)$ reconstruction error $\\varepsilon"
           "(\\ell_{\\max})=\\lVert f-S_{\\ell_{\\max}}f\\rVert/\\lVert f\\rVert$ "
           "(area-weighted, log--log) for the EGM2008 geoid and the IGRF-13 "
           "radial field $B_r$. The error is the exact area-weighted Parseval "
           "ratio, so each curve is deterministic; the legend gives the decay "
           "exponent $\\hat{p}=(1-\\alpha)/2$ from a weighted fit of the "
           "degree-variance spectrum, with its standard error. Black dashed "
           "lines are the $\\varepsilon$-OLS cross-check fits; squares mark "
           "where $\\varepsilon$ crosses 10, 5 and 1%. The magnetic field "
           "decays far more steeply and is intrinsically limited to "
           "$\\ell\\le13$.")
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure 3 -- power (degree-variance) spectra
# --------------------------------------------------------------------------- #
def figure3_spectra(grav_res, mag_res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) gravity geoid degree variance (m^2)
    g = grav_res["field"]
    Pl = grav_res["curve"]["Pl"]
    ls = np.arange(len(Pl))
    m = Pl > 0
    axes[0].plot(ls[m], Pl[m], "-", color=g["color"], lw=1.8)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Spherical-harmonic degree  $\ell$")
    axes[0].set_ylabel(r"Geoid degree variance  $\sigma_\ell^2$  (m$^2$)")
    axes[0].set_title("(a) Gravity: EGM2008 geoid")
    axes[0].xaxis.set_major_formatter(mticker.ScalarFormatter())

    # (b) magnetic Lowes-Mauersberger spectrum (nT^2)
    f = mag_res["field"]
    ls_m, Rl = f["lowes"]
    m = (ls_m >= 1) & (Rl > 0)
    axes[1].plot(ls_m[m], Rl[m], "o-", color=f["color"], lw=1.8, ms=4)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Spherical-harmonic degree  $\ell$")
    axes[1].set_ylabel(r"Lowes--Mauersberger  $R_\ell$  (nT$^2$)")
    axes[1].set_title("(b) Magnetic: IGRF-13 main field")
    axes[1].set_xticks(range(1, 14, 2))

    fig.suptitle("Power spectra (degree variance) of the two fields",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    paths = _save(fig, "fig3_power_spectra")
    cap = ("Figure 3. Degree-variance power spectra. (a) EGM2008 geoid degree "
           "variance $\\sigma_\\ell^2$ (m$^2$) versus degree $\\ell$ (log--log); "
           "power decreases slowly, approximately following Kaula's rule, so "
           "much signal lives at high degree. (b) IGRF-13 main-field "
           "Lowes--Mauersberger spectrum $R_\\ell=(\\ell+1)\\sum_m(g_\\ell^{m2}"
           "+h_\\ell^{m2})$ (nT$^2$); the degree-1 dipole dominates and power "
           "falls steeply to the model limit $\\ell=13$. These shapes explain "
           "the contrasting truncation-error decay in Figure 2.")
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure 4 -- compression-ratio / error Pareto curve
# --------------------------------------------------------------------------- #
def figure4_pareto(results):
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    for r in results:
        f = r["field"]
        sec = f.get("secondary", False)
        lm = np.asarray(r["curve"]["lmax"], float)
        eg = r["curve"]["eps_grid"]
        ncoef = (lm + 1) ** 2                            # absolute storage cost
        pos = eg > 0
        lab = f["label"] + ("  [secondary]" if sec else "")
        style = "--" if sec else "o-"
        ax.plot(ncoef[pos], eg[pos], style, color=f["color"],
                ms=0 if sec else 4, lw=1.4 if sec else 1.8,
                alpha=0.9 if sec else 1.0, label=lab)
        if not sec:
            cr = S.crossing_degrees(lm, eg, C.MARKER_THRESHOLDS)
            for t in C.MARKER_THRESHOLDS:
                l_at = cr[t]
                if l_at is not None:
                    ax.plot([(l_at + 1) ** 2], [t], marker="s", color=f["color"],
                            ms=7, mec="k", mew=0.6, zorder=5)
    for t in C.MARKER_THRESHOLDS:
        ax.axhline(t, color=C.COLOR_THRESH, lw=0.8, ls=":", alpha=0.6)
        ax.text(0.995, t, " %g%%" % (t * 100), color=C.COLOR_THRESH,
                fontsize=9, va="bottom", ha="right",
                transform=ax.get_yaxis_transform())

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of coefficients retained  $N=(\ell_{\max}+1)^2$"
                  "   (storage cost)")
    ax.set_ylabel(r"Relative $L^2(S^2)$ error  $\varepsilon$")
    ax.set_title("Compression / accuracy trade-off (Pareto curve)")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.text(0.985, 0.97,
            "left = fewer coefficients (cheaper)\n"
            "squares = 15 / 10 / 5 / 2.5 / 1% error\n"
            "compression ratio = $N/(\\ell_\\mathrm{ref}{+}1)^2$",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    paths = _save(fig, "fig4_compression_pareto")
    cap = ("Figure 4. Accuracy/compression trade-off. Relative error "
           "$\\varepsilon$ versus the number of coefficients retained "
           "$N=(\\ell_{\\max}+1)^2$ (absolute storage cost; both axes log); the "
           "compression ratio is $N/(\\ell_{\\mathrm{ref}}+1)^2$ relative to the "
           "reference ($\\ell_{\\mathrm{ref}}=%d$ gravity, %d magnetic). Squares "
           "mark the 15/10/5/2.5/1%% error levels. In absolute terms the IGRF-13 "
           "main field is far cheaper -- about %d coefficients reach 1%% error "
           "versus roughly %d for the geoid -- because it is intrinsically "
           "band-limited to $\\ell\\le13$; the crustal field (secondary) "
           "compresses poorly (it needs nearly all of its coefficients)."
           % (C.GRAV_LREF, C.MAG_LREF,
              (int(np.ceil(results[1]['thresholds'][0.01])) + 1) ** 2,
              (int(np.ceil(results[0]['thresholds'][0.01])) + 1) ** 2))
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure 5 -- residual (lost) field at the l_max where eps ~ 5%
# --------------------------------------------------------------------------- #
def figure5_residual(grav_res, precomputed=None):
    """Render figure 5.

    Parameters
    ----------
    grav_res : dict
        Gravity result dict (must contain 'field' with 's', 'lref', 'map_lmax'
        and 'thresholds').
    precomputed : dict, optional
        If provided, must have keys 'lons', 'lats', 'resid', 'l_use', 'lref'.
        When supplied these arrays are used directly instead of calling
        map_arrays() on grav_res['field']['s'].
    """
    grav = grav_res["field"]
    if precomputed is not None:
        lons = precomputed["lons"]
        lats = precomputed["lats"]
        resid = precomputed["resid"]
        l_use = precomputed["l_use"]
        lref = precomputed["lref"]
    else:
        s = grav["s"]
        l5 = grav_res["thresholds"][0.05]
        l_use = int(round(l5))
        lref = grav["lref"]
        lons, lats, ref = map_arrays(s, lref, grav["map_lmax"])
        _, _, trunc = map_arrays(s, l_use, grav["map_lmax"])
        resid = ref - trunc
    vmax = float(np.ceil(np.percentile(np.abs(resid), 99.5)))

    fig = plt.figure(figsize=(9.5, 5.2))
    ax = _geoaxes(fig, fig.add_gridspec(1, 1)[0, 0])
    mesh = _draw_map(ax, lons, lats, resid, vmax, C.DIVERGING_CMAP,
                     title=(r"Geoid signal lost by truncating at "
                            r"$\ell_{\max}=%d$  ($\varepsilon\approx5\%%$)"
                            % l_use))
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.025,
                        pad=0.03, extend="both")
    cbar.set_label("Geoid residual $N_{\\mathrm{ref}}-N_{\\ell_{\\max}}$ (m)")
    paths = _save(fig, "fig5_gravity_residual")
    cap = ("Figure 5. Spatial pattern of the geoid signal lost by spectral "
           "truncation. Residual $N_{\\mathrm{ref}}-N_{\\ell_{\\max}}$ (m) "
           "between the reference EGM2008 geoid ($\\ell_{\\mathrm{ref}}=%d$) and "
           "its truncation at $\\ell_{\\max}=%d$, the degree where the relative "
           "error reaches about 5%%. The discarded short-wavelength power is "
           "visibly organised along tectonically active plate boundaries -- "
           "subduction zones and trenches (e.g.\\ the Andes, Indonesia, the "
           "western Pacific) and active orogens such as the Himalaya -- where "
           "short-wavelength geoid anomalies are largest because the supporting "
           "masses are only partially isostatically compensated. We report this "
           "spatial association qualitatively and do not quantify a "
           "geoid--topography correlation here." % (lref, l_use))
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure A -- rate-distortion / codec bit-efficiency gap (CORE headline figure)
# --------------------------------------------------------------------------- #
_FIELDA = [("gravity", "Gravity geoid (EGM2008)", C.COLOR_GRAVITY),
           ("magnetic", "Main field $B_r$ (IGRF-13)", C.COLOR_MAGNETIC),
           ("crustal", "Crustal $B_r$ (NGDC-720)", C.COLOR_LITHO)]


def figureA_codec(results_dir=None):
    """Figure A: storage in bits vs target relative error for each field --
    float32 coefficient storage, the rate-distortion floor (reverse water-fill),
    and two realized coders (greedy quantizer, entropy-coded).

    Every plotted number is read verbatim from results/water_fill.json
    (float32 + RD floor) and results/codec_gap.json (greedy + entropy)."""
    import json
    if results_dir is None:
        results_dir = C.DIR_RESULTS
    with open(os.path.join(results_dir, "water_fill.json")) as fh:
        wf = json.load(fh)
    with open(os.path.join(results_dir, "codec_gap.json")) as fh:
        cg = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=False)
    ratios_1pct = {}
    for ax, (key, label, color) in zip(axes, _FIELDA):
        wrows = wf[key]
        crows = cg[key]
        eps = np.array([r["eps"] for r in wrows])
        b_float = np.array([r["B_float32_bits"] for r in wrows])
        b_wf = np.array([r["B_WF_bits"] for r in wrows])
        # cross-check the two JSONs agree on the shared quantities
        assert np.allclose(eps, [r["eps"] for r in crows]), "%s eps grid mismatch" % key
        assert np.allclose(b_wf, [r["B_WF"] for r in crows], rtol=1e-6), \
            "%s B_WF mismatch between water_fill.json and codec_gap.json" % key
        b_real = np.array([r["B_real"] for r in crows])
        b_ent = np.array([r["B_entropy"] for r in crows])

        ax.plot(eps, b_float, "s-", color="0.35", lw=1.6, ms=5, label="float32")
        ax.plot(eps, b_real, "^-", color=color, lw=1.6, ms=5,
                label="greedy quantizer")
        ax.plot(eps, b_ent, "v--", color=color, lw=1.4, ms=5, alpha=0.85,
                label="entropy-coded")
        ax.plot(eps, b_wf, "o-", color="k", lw=1.8, ms=4,
                label="rate-distortion floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()                       # accuracy improves left -> right
        ax.set_xlabel(r"Target relative error  $\varepsilon$")
        ax.set_title(label, fontsize=12)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        # record the eps=1% ratios for the (factual) caption
        i1 = int(np.argmin(np.abs(eps - 0.01)))
        ratios_1pct[key] = (b_float[i1] / b_wf[i1], b_real[i1] / b_wf[i1],
                            b_ent[i1] / b_wf[i1])
    axes[0].set_ylabel("Storage (bits)")
    fig.suptitle("Storage in bits vs target accuracy: float32, realized coders, "
                 "and the rate-distortion floor", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = _save(fig, "figA_codec_bits")

    g = ratios_1pct["gravity"]; m = ratios_1pct["magnetic"]; c = ratios_1pct["crustal"]
    cap = ("Figure A. Storage in bits versus target relative error $\\varepsilon$ "
           "(both axes logarithmic; accuracy increases left to right) for the "
           "geoid, main field, and crustal field. Each panel shows four "
           "quantities: float32 coefficient storage, the rate-distortion floor "
           "$B_{\\mathrm{WF}}$ (reverse water-filling of independent Gaussian "
           "coefficients), and two realized coders -- a greedy scalar quantizer "
           "($B_{\\mathrm{real}}$) and entropy coding of the quantized symbols "
           "($B_{\\mathrm{entropy}}$). All values are read from "
           "\\texttt{results/water\\_fill.json} (float32, floor) and "
           "\\texttt{results/codec\\_gap.json} (greedy, entropy). At "
           "$\\varepsilon=1\\%%$ the float32 storage is %.1f, %.1f and %.1f times "
           "the floor for the geoid, main field and crustal field; the greedy "
           "coder is %.2f, %.2f and %.2f times the floor and the entropy coder "
           "%.2f, %.2f and %.2f times the floor (values below~1 indicate the "
           "entropy of the quantized symbols falls under the Gaussian "
           "rate-distortion bound). "
           "[PALAASH: interpretation of the float32-vs-floor gap and the "
           "coder-vs-floor gap goes here.]"
           % (g[0], m[0], c[0], g[1], m[1], c[1], g[2], m[2], c[2]))
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure B -- downward-continuation compressibility of the main field (CORE)
# --------------------------------------------------------------------------- #
def figureB_radius(results_dir=None):
    """Figure B: 1% truncation degree and decay slope of the IGRF main field as
    it is analytically continued from Earth's surface to the core-mantle
    boundary.  Every number from results/radius_compressibility.csv."""
    import csv
    if results_dir is None:
        results_dir = C.DIR_RESULTS
    aor, rkm, l1, slope = [], [], [], []
    with open(os.path.join(results_dir, "radius_compressibility.csv")) as fh:
        for row in csv.DictReader(fh):
            rkm.append(float(row["r_km"])); aor.append(float(row["a_over_r"]))
            l1.append(float(row["l_at_1pct"])); slope.append(float(row["decay_slope"]))
    aor = np.array(aor); l1 = np.array(l1); slope = np.array(slope)

    fig, ax1 = plt.subplots(figsize=(8.6, 5.6))
    c1, c2 = C.COLOR_MAGNETIC, C.COLOR_GRAVITY
    ax1.plot(aor, l1, "o-", color=c1, lw=1.8, ms=5)
    ax1.set_xlabel(r"Continuation ratio  $a/r$  ($a=6371.2$ km surface; "
                   r"$a/r{=}1.83$ at the CMB)")
    ax1.set_ylabel(r"Degree at $\varepsilon=1\%$  ($\ell_{1\%}$)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.plot(aor, slope, "s--", color=c2, lw=1.6, ms=5)
    ax2.set_ylabel(r"Decay slope  $\hat{p}$", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.grid(False)
    ax1.set_title("Downward-continuation compressibility of the IGRF main field")
    paths = _save(fig, "figB_radius")

    cap = ("Figure B. Compressibility of the IGRF-13 radial main field as it is "
           "analytically continued downward from Earth's surface ($a/r=1$) to the "
           "core--mantle boundary ($a/r\\approx1.83$, $r\\approx3480$ km). Left "
           "axis (circles): the degree $\\ell_{1\\%%}$ at which the relative error "
           "reaches 1\\%%, rising from %.1f at the surface to %.1f at the CMB. "
           "Right axis (squares): the log--log decay slope $\\hat{p}$, changing "
           "from $%.2f$ to $%.2f$. All values from "
           "\\texttt{results/radius\\_compressibility.csv}. Caveat: this is the "
           "analytic continuation of the degree-13 IGRF model, so the high-degree "
           "amplification acts on coefficients near the model's noise floor; it is "
           "a property of the model field, not a measurement at depth. "
           "[PALAASH: interpretation of the depth--compressibility trend goes "
           "here.]"
           % (l1[0], l1[-1], slope[0], slope[-1]))
    return paths, cap


# --------------------------------------------------------------------------- #
# Figure D -- planetary geoid decay curves (BREADTH; neutral)
# --------------------------------------------------------------------------- #
_FIELDD = [("earth", "Earth geoid (EGM2008, $\\ell_{\\mathrm{ref}}{=}360$)",
            C.COLOR_GRAVITY, "o-"),
           ("moon", "Moon geoid (GRGM1200B, $\\ell_{\\mathrm{ref}}{=}360$)",
            C.COLOR_MAGNETIC, "s-"),
           ("mars", "Mars areoid (GMM3, $\\ell_{\\mathrm{ref}}{=}90$)",
            C.COLOR_LITHO, "^-")]


def figureD_planetary(results_dir=None):
    """Figure D: exact per-degree Parseval error eps(L) for the Earth, Moon and
    Mars geoids.  Earth and Moon share lref=360 (comparable); Mars is referenced
    to lref=90 and is NOT comparable on coefficient count.  Every number from
    results/planetary_curves.csv (which reproduces the probe crossings)."""
    import csv
    if results_dir is None:
        results_dir = C.DIR_RESULTS
    data = {}
    with open(os.path.join(results_dir, "planetary_curves.csv")) as fh:
        for row in csv.DictReader(fh):
            data.setdefault(row["body"], ([], []))
            data[row["body"]][0].append(int(row["lmax"]))
            data[row["body"]][1].append(float(row["eps"]))

    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    for key, label, color, style in _FIELDD:
        L = np.array(data[key][0], float); e = np.array(data[key][1], float)
        m = (L >= 1) & (e > 0)
        ax.plot(L[m], e[m], style, color=color, lw=1.8, ms=3, label=label)
    ax.axhline(0.01, color=C.COLOR_THRESH, lw=0.9, ls=":", alpha=0.8)
    ax.text(0.995, 0.01, " 1%", color=C.COLOR_THRESH, fontsize=9,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Truncation degree  $\ell_{\max}$")
    ax.set_ylabel(r"Relative $L^2(S^2)$ error  $\varepsilon$")
    ax.set_title("Planetary geoid truncation error (exact Parseval)")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    paths = _save(fig, "figD_planetary")

    cap = ("Figure D. Exact relative $L^2(S^2)$ truncation error "
           "$\\varepsilon(\\ell_{\\max})=\\sqrt{\\sum_{\\ell>\\ell_{\\max}}"
           "\\sigma_\\ell^2/\\sum_\\ell\\sigma_\\ell^2}$ for the geoids of Earth "
           "(EGM2008), the Moon (GRGM1200B) and Mars (GMM3), on log--log axes; "
           "the dotted line marks the 1\\%% level. Earth and the Moon are both "
           "referenced to $\\ell_{\\mathrm{ref}}=360$ and are directly comparable. "
           "Mars uses a lower-degree model referenced to "
           "$\\ell_{\\mathrm{ref}}=90$ and is therefore NOT comparable on "
           "coefficient count with Earth or the Moon. Values from "
           "\\texttt{results/planetary\\_curves.csv}. "
           "[PALAASH: methodological point -- the decay slope (a local rate) and "
           "the coefficient count at a fixed tolerance (a level-dependent "
           "threshold) are distinct summaries: the Moon's geoid is steeper than "
           "Earth's yet crosses 1\\% at a higher degree because its normalized "
           "error exceeds Earth's at every degree 5--359.]")
    return paths, cap


# --------------------------------------------------------------------------- #
# captions writer
# --------------------------------------------------------------------------- #
def write_captions(captions):
    path = os.path.join(C.DIR_FIGURES, "captions.md")
    with open(path, "w") as fh:
        fh.write("# Figure captions\n\n")
        for name, cap in captions:
            fh.write("## %s\n\n%s\n\n" % (name, cap))
    return path


# --------------------------------------------------------------------------- #
# Aux-figure rendering (A/B/D) + copy into the manuscript figures dir
# --------------------------------------------------------------------------- #
_TEX_FIGDIR = os.path.join(C.ROOT, "geoid", "geoid_paper_arxiv_overleaf",
                           "figures")


def _copy_to_tex(paths):
    """Copy rendered PDF(s) next to main.tex so \\includegraphics resolves."""
    import shutil
    if not os.path.isdir(_TEX_FIGDIR):
        return
    for p in paths:
        if p.endswith(".pdf"):
            shutil.copy(p, os.path.join(_TEX_FIGDIR, os.path.basename(p)))


def render_aux_figures(results_dir=None, copy_tex=True):
    """Render figures A/B/D purely from the results JSON/CSV files (no model
    download, no SHCoeffs).  Each figure is skipped (with a message) if its
    input file is missing, so this never breaks the core pipeline.  Returns the
    list of (name, caption) pairs that were produced."""
    if results_dir is None:
        results_dir = C.DIR_RESULTS
    plan = [
        ("Figure A", "figA_codec_bits", figureA_codec,
         ["water_fill.json", "codec_gap.json"]),
        ("Figure B", "figB_radius", figureB_radius,
         ["radius_compressibility.csv"]),
        ("Figure D", "figD_planetary", figureD_planetary,
         ["planetary_curves.csv"]),
    ]
    captions = []
    for name, base, fn, inputs in plan:
        missing = [f for f in inputs
                   if not os.path.isfile(os.path.join(results_dir, f))]
        if missing:
            print("   - %s SKIP (missing %s -- run the producing script)"
                  % (base, ", ".join(missing)))
            continue
        paths, cap = fn(results_dir)
        if copy_tex:
            _copy_to_tex(paths)
        captions.append((name, cap))
        print("   - %s PASS" % os.path.basename(paths[0]))
    return captions


def render_from_cache(cache_path, outdir=None):
    """Render ALL figures with no data download: figures 1-5 from the pickled
    results cache, figures A/B/D from the results JSON/CSV files."""
    import pickle
    if outdir is not None:
        C.DIR_FIGURES = outdir
    os.makedirs(C.DIR_FIGURES, exist_ok=True)
    with open(cache_path, "rb") as fh:
        cache = pickle.load(fh)
    grav_stub = {
        "map_degrees": cache["map_degrees"], "label": cache["grav_label"],
        "lref": cache["grav_lref"], "map_lmax": cache["grav_map_lmax"],
        "s": None,
    }
    rs = cache["results_stripped"]
    caps = []
    p, c = figure1_reconstructions(grav_stub, precomputed_maps=cache["fig1_maps"])
    _copy_to_tex(p); caps.append(("Figure 1", c)); print("   -", os.path.basename(p[0]), "PASS")
    p, c = figure2_error_curves(rs); _copy_to_tex(p); caps.append(("Figure 2", c)); print("   -", os.path.basename(p[0]), "PASS")
    p, c = figure3_spectra(rs[0], rs[1]); _copy_to_tex(p); caps.append(("Figure 3", c)); print("   -", os.path.basename(p[0]), "PASS")
    p, c = figure4_pareto(rs); _copy_to_tex(p); caps.append(("Figure 4", c)); print("   -", os.path.basename(p[0]), "PASS")
    p, c = figure5_residual(rs[0], precomputed=cache["fig5_data"]); _copy_to_tex(p); caps.append(("Figure 5", c)); print("   -", os.path.basename(p[0]), "PASS")
    caps += render_aux_figures()
    write_captions(caps)
    return caps


# --------------------------------------------------------------------------- #
# CLI entry point: figures.py --from-cache PATH   (figs 1-5 from pkl + A/B/D)
#                  figures.py --aux-only          (only A/B/D from results files)
# --------------------------------------------------------------------------- #
def main():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Render figures with no data download. --from-cache renders all of "
            "figures 1-5 (from a .pkl results cache) plus A/B/D (from results "
            "JSON/CSV); --aux-only renders just A/B/D from the results files."
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--from-cache", metavar="PATH",
        help="Path to the .pkl cache file written by make_figures.py "
             "--save-cache. Cache files use Python pickle; load only caches you "
             "generated yourself (do not load untrusted .pkl files).",
    )
    g.add_argument(
        "--aux-only", action="store_true",
        help="Render only figures A/B/D from results/*.json and results/*.csv.",
    )
    p.add_argument(
        "--outdir", default=C.DIR_FIGURES, metavar="PATH",
        help="Output directory for figure files. Default: %(default)s",
    )
    args = p.parse_args()

    C.DIR_FIGURES = args.outdir
    os.makedirs(C.DIR_FIGURES, exist_ok=True)

    if args.aux_only:
        render_aux_figures()
        return 0

    print("Loading cache from %s ..." % args.from_cache)
    render_from_cache(args.from_cache, outdir=args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
