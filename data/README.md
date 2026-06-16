# data/

This study does **not** vendor the raw model files. They are downloaded on first
run and cached by `pyshtools` under `~/.cache/pyshtools/`. pyshtools uses the
`pooch` library to manage downloads; the default cache directory is
`~/.pyshtools/` (or `pooch.os_cache('pyshtools')`, which resolves to
`~/.cache/pyshtools/` on Linux/macOS). Downloaded files persist between runs;
delete this directory to force a fresh download.

- **EGM2008** (gravity), ~250 MB, from ICGEM. Accessor:
  `pyshtools.datasets.Earth.EGM2008(lmax=360)` — returns `SHGravRealCoeffs`.
  Calibrated commission errors are loaded automatically and accessed via the
  `.errors` attribute; do **not** pass `errors='calibrated'` as a keyword
  argument (pyshtools 4.14.1 auto-loads them; passing that kwarg raises
  `TypeError`). Errors are used as the gravity noise floor in the
  effective-bandlimit analysis (Component 1 / task 3.4). Pavlis, Holmes, Kenyon &
  Factor (2012), JGR Solid Earth 117, B04406, doi:10.1029/2011JB008916.
  Ellipsoid: `boule.WGS84`.
- **IGRF-13** (magnetic main field), ~40 kB, from NOAA NCEI. Accessor:
  `pyshtools.datasets.Earth.IGRF_13(lmax=13, year=2020.0)` — returns
  `SHMagRealCoeffs`. Time-variable 1900–2025 (5-yr epochs). **IMPORTANT:
  degrees 11–13 are zero for epochs before 2000.0** (the model extended from
  10-degree to 13-degree in 2000); the temporal analysis (task 3.3) accounts for
  this by detecting the last non-zero degree per epoch. The secular variation is
  used as the magnetic noise proxy (3.4). Alken et al. (2021), Earth Planets
  Space 73, 49.
- **NGDC-720 V3** (crustal magnetic, secondary), ~3 MB, from NOAA NCEI. Accessor:
  `pyshtools.datasets.Earth.NGDC_720_V3(lmax=133)` — degrees 1–15 are zero by
  construction (crustal model).

**Planetary gravity (task 3.1)** — accessors confirmed against the installed
pyshtools 4.14.1 (note: Mars `MRO120D` does **not** exist; GMM3 is used):

- **Moon GRGM1200B** (GRAIL), native degree 1200, loaded at lmax=360 for this
  study via `pyshtools.datasets.Moon.GRGM1200B(lmax=360)`; selenoid relative to
  `boule.Moon2015` (sphere, f=0, R=1737151 m). Goossens et al. (2020), JGR
  Planets 125, e2019JE006086, doi:10.1029/2019JE006086. Ellipsoid: Wieczorek
  (2015), Treatise on Geophysics 2nd ed. 10.05.
- **Mars GMM3**, native degree 120, loaded at lmax=90 via
  `pyshtools.datasets.Mars.GMM3(lmax=90)`; areoid relative to `boule.Mars2009`
  (a=3395428 m, f=0.005228). NOTE: `MRO120D` does **not** exist in pyshtools
  4.14.1; `GMM3` is the canonical static Mars gravity model used here. `MRO120F`
  (Konopliv et al. 2020) is an alternative JPL model but is not used. Genova et
  al. (2016), Icarus 272, 228–245, doi:10.1016/j.icarus.2016.02.050. Ellipsoid:
  Ardalan, Karimi & Grafarend (2009), Earth Moon Planets 106, 1–13,
  doi:10.1007/s11038-009-9342-7.

**Ellipsoid accessors (boule)**:

- `boule.WGS84` — Earth reference ellipsoid
- `boule.Moon2015` — Sphere (f=0, R=1737151 m)
- `boule.Mars2009` — Ellipsoid (a=3395428 m, f=0.005228)

**pyshtools version note**: pyshtools 4.14.1 exposes no `__version__` attribute
(returns `None` from `getattr`); use
`importlib.metadata.version('pyshtools')` or check the conda env pin. The conda
env pin is `pyshtools==4.14.1` (recorded in requirements.txt).

Running `python make_figures.py` (Earth) / `python planetary.py` (Moon, Mars) /
`python functionals.py` re-downloads anything missing. Per-run download logs are
written here but are git-ignored.
