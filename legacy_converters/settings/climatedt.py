"""
ClimateDT (IFS-NEMO) default conversion settings.

ClimateDT output is natively on HEALPix level 7 NESTED (nside=128, ~50 km/pixel).
No regridding is needed — data is directly packaged into the EOPF zarr format.

The PSFResampler can optionally be applied for a sphere → WGS84 ellipsoid
correction (the source coordinates are spherical, the target grid is on the
WGS84 ellipsoid). This adds ~5-10 s per timestep.

Output group layout:
  measurements/sfc/     – surface IFS fields
  measurements/ocean/   – NEMO 3D ocean fields (one sub-group per depth level)
"""

# ── HEALPix grid constants ────────────────────────────────────────────────────
_HEALPIX_L7_WGS84 = {
    "refinement_level": 7,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}
_HEALPIX_L7_CHUNKS = {
    "refinement_level": 2,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}

# ── Variable metadata ─────────────────────────────────────────────────────────
# (MARS param_id, unit, long_name)
CDT_SFC_VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "sp": ("134", "Pa", "Surface pressure"),
    "u10": ("165", "m s-1", "10 metre U wind component"),
    "v10": ("166", "m s-1", "10 metre V wind component"),
    "t2m": ("167", "K", "2 metre temperature"),
    "d2m": ("168", "K", "2 metre dewpoint temperature"),
}
CDT_OCEAN_VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "avg_thetao": ("263501", "K", "Ocean potential temperature"),
}

# ── ConvertSettings dict ──────────────────────────────────────────────────────
CDT_SFC_CONVERT_SETTINGS: dict = {
    "group_settings": {
        "measurements/sfc": {
            "healpix": _HEALPIX_L7_WGS84,
            "chunk": {
                "method": "healpix_cell_dense",
                "healpix": _HEALPIX_L7_CHUNKS,
                "chunk_buffer_width": 0.0,
            },
            "resampler": {
                "name": "nearest",
                "init_params": {},
                "resample_params": {},
            },
        },
    },
}

# ── Polytope API constants ────────────────────────────────────────────────────
POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"
POLYTOPE_COLLECTION = "destination-earth"

CDT_CLASS = "d1"
CDT_DATASET = "climate-dt"
CDT_EXPERIMENT = "SSP3-7.0"
CDT_EXPVER = "0001"
CDT_GENERATION = "1"
CDT_MODEL = "IFS-NEMO"
CDT_RESOLUTION = "standard"
CDT_STREAM = "clte"
CDT_TYPE = "fc"

CDT_DEFAULT_SFC_PARAMS = "134/165/166/167/168"
CDT_DEFAULT_OCEAN_PARAM = "263501"
