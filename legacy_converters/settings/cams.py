"""
CAMS EAC4 default conversion settings.

CAMS reanalysis aerosol optical depth on a regular 0.75° lat/lon grid
projected to HEALPix level 6 (~100 km/pixel). The default resampler is the
PSFResampler (same engine as ERA5), tuned for the regular grid; nearest-neighbour
binning remains available as an alternative (CAMSConverter(method="nn")).

Why level 6 and not 7?
  CAMS 0.75° ≈ 83 km resolution → 115,680 source points.
  HEALPix level 7 has 196,608 pixels at ~50 km — finer than the source grid.
  Level 6 (49,152 pixels at ~100 km) matches the native resolution honestly.
  Both methods now give full coverage (0 NaN) at level 6 and 7; level 6 is kept
  as the default because going finer is upsampling beyond the source resolution.

Resampler choice (PSF vs nearest-neighbour):
  PSFResampler was historically avoided on this regular grid because its defaults
  (threshold=0.5, lam=0.0) produced NaN holes at level 7 and Gibbs ringing
  (negative AOD) at the edges of dust/fire plumes. Two parameters fix both:
    - threshold=0.01 → every HEALPix cell is covered (0 NaN, even at level 7);
    - lam=5.0 (Tikhonov damping) → no ringing (0 negative AOD).
  A final clip(0, None) is the physical safety net. Validated on the 6 CAMS
  variables: 0 NaN, 0 negative, correlation > 0.98 with nearest-neighbour binning.
  Nearest-neighbour binning (method="nn") is exact and faster; the ~1% of empty
  equatorial pixels are filled from HEALPix ring neighbours.

Output group layout:
  measurements/aod/   – all AOD variables
"""

# ── HEALPix grid constants ────────────────────────────────────────────────────
_HEALPIX_L6_WGS84 = {
    "refinement_level": 6,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}
_HEALPIX_L6_CHUNKS = {
    "refinement_level": 2,  # → 192 parent cells, 256 child pixels each
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}

# ── Variable metadata ─────────────────────────────────────────────────────────
# (ADS variable name, unit, long_name)
CAMS_VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "aod550": ("total_aerosol_optical_depth_550nm", "1", "Total AOD at 550 nm"),
    "ssaod550": ("sea_salt_aerosol_optical_depth_550nm", "1", "Sea salt AOD at 550 nm"),
    "duaod550": ("dust_aerosol_optical_depth_550nm", "1", "Dust AOD at 550 nm"),
    "omaod550": (
        "organic_matter_aerosol_optical_depth_550nm",
        "1",
        "Organic matter AOD at 550 nm",
    ),
    "bcaod550": (
        "black_carbon_aerosol_optical_depth_550nm",
        "1",
        "Black carbon AOD at 550 nm",
    ),
    "suaod550": ("sulphate_aerosol_optical_depth_550nm", "1", "Sulphate AOD at 550 nm"),
}

# ── ConvertSettings dict ──────────────────────────────────────────────────────
CAMS_CONVERT_SETTINGS: dict = {
    "group_settings": {
        "measurements/aod": {
            "healpix": _HEALPIX_L6_WGS84,
            "chunk": {
                "method": "healpix_cell_dense",
                "healpix": _HEALPIX_L6_CHUNKS,
                "chunk_buffer_width": 0.0,
            },
            "resampler": {
                "name": "psf",
                "init_params": {"threshold": 0.01},
                "resample_params": {"lam": 5.0},
            },
        },
    },
}

# ── PSFResampler tuning (regular grid → HEALPix) ──────────────────────────────
# threshold lowered from the ERA5 default (0.5) so every HEALPix cell is covered;
# lam is the Tikhonov damping that suppresses Gibbs ringing (negative AOD).
CAMS_PSF_THRESHOLD = 0.01
CAMS_PSF_LAM = 5.0

# ── ADS API constants ─────────────────────────────────────────────────────────
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
ADS_DATASET = "cams-global-reanalysis-eac4"
CAMS_GRID_DEG = 0.75  # native resolution of EAC4
