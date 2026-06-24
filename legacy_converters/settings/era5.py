"""
ERA5 default conversion settings.

ERA5 reanalysis surface fields on native Gaussian grids (N256 / N320)
projected to HEALPix level 7 (~50 km/pixel) using PSF resampling.

MARS strategy: two separate requests to handle variable availability:
  ① stream=enda / type=an / N256 – ensemble surface fields
  ② stream=oper / type=an / N320 – additional deterministic fields

Output group layout (matches the framework's measurements structure):
  measurements/enda/   – ensemble fields  (t2m, u10, v10, sp, d2m, skt)
  measurements/oper/   – deterministic fields  (tcwv, mslp, tcc, sst, sic, …)
"""

# ── HEALPix grid shared constants ─────────────────────────────────────────────
_HEALPIX_L7_WGS84 = {
    "refinement_level": 7,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}
_HEALPIX_L7_CHUNKS = {  # parent cells = chunks
    "refinement_level": 2,  # → 12 × 4² = 192 parent cells
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}

# ── Variable metadata ─────────────────────────────────────────────────────────
# (GRIB param_id, unit, long_name)
ERA5_ENDA_VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "t2m": ("167.128", "K", "2 metre temperature"),
    "u10": ("165.128", "m s-1", "10 metre U wind component"),
    "v10": ("166.128", "m s-1", "10 metre V wind component"),
    "sp": ("134.128", "Pa", "Surface pressure"),
    "d2m": ("168.128", "K", "2 metre dewpoint temperature"),
    "skt": ("235.128", "K", "Skin temperature"),
}
ERA5_OPER_VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "tcwv": ("137.128", "kg m-2", "Total column water vapour"),
    "mslp": ("151.128", "Pa", "Mean sea level pressure"),
    "tcc": ("164.128", "1", "Total cloud cover"),
    "sst": ("34.128", "K", "Sea surface temperature"),
    "sic": ("31.128", "1", "Sea ice area fraction"),
    "sd": ("141.128", "m", "Snow depth (water equivalent)"),
    "swvl1": ("39.128", "m3 m-3", "Volumetric soil water layer 1"),
    "asn": ("32.128", "1", "Snow albedo"),
    "tco3": ("206.128", "kg m-2", "Total column ozone"),
}
ERA5_ALL_VARIABLE_META = {**ERA5_ENDA_VARIABLE_META, **ERA5_OPER_VARIABLE_META}

# ── ConvertSettings dict ──────────────────────────────────────────────────────
ERA5_CONVERT_SETTINGS: dict = {
    "group_settings": {
        # ── Ensemble surface fields (N256) – PSF resampling ──────────────────
        "measurements/enda": {
            "healpix": _HEALPIX_L7_WGS84,
            "chunk": {
                "method": "healpix_cell_dense",
                "healpix": _HEALPIX_L7_CHUNKS,
                "chunk_buffer_width": 0.0,
            },
            "resampler": {
                "name": "psf",
                "init_params": {"threshold": 0.5},
                "resample_params": {"lam": 0.0},
            },
        },
        # ── Deterministic surface fields (N320) – PSF resampling ─────────────
        "measurements/oper": {
            "healpix": _HEALPIX_L7_WGS84,
            "chunk": {
                "method": "healpix_cell_dense",
                "healpix": _HEALPIX_L7_CHUNKS,
                "chunk_buffer_width": 0.0,
            },
            "resampler": {
                "name": "psf",
                "init_params": {"threshold": 0.5},
                "resample_params": {"lam": 0.0},
            },
        },
    },
}

# ── CDS API constants ─────────────────────────────────────────────────────────
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_DATASET = "reanalysis-era5-complete"
ERA5_ENDA_GRID = "N256"
ERA5_OPER_GRID = "N320"
DEFAULT_NUMBERS = list(range(10))  # ensemble members 0-9
