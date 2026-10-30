"""
Copernicus DEM GLO-30 default conversion settings.

Copernicus DEM GLO-30 (COP-DEM_GLO-30-DGED) surface heights on their native
1 arc-second lon/lat grid, projected to HEALPix level 18 (~25 m/pixel) using
bilinear resampling, with a pyramid of block means down to level 8 (~25 km)
so that a viewer can show the whole area before zooming in.

Input group layout (one area of interest staged from the source dataset, see
the example notebook):
  dem/glo30/   – ``dsm`` on the ``lat`` / ``lon`` grid

Output group layout:
  dem/         – multiscale group (``multiscales`` metadata of the pyramid)
  dem/18/      – ``dsm`` resampled from the input
  dem/17/ … dem/8/ – ``dsm`` block means of level 18

The level-18 data is sharded: each shard (one object) holds the level-18 cells
of one level-6 cell, in inner chunks of one level-8 cell (4**10 cells, as in the
published GRID4EARTH DEM), which readers fetch by byte range. A shard is also
the unit of work of the conversion.
"""

# ── HEALPix grid shared constants ─────────────────────────────────────────────
_HEALPIX_L18_WGS84 = {
    "refinement_level": 18,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}
_HEALPIX_L6_SHARDS = {  # parent cells = shards (4**12 cells each)
    "refinement_level": 6,
    "indexing_scheme": "nested",
    "ellipsoid": {"name": "wgs84"},
}
_L8_INNER_CHUNK = 4 ** (18 - 8)  # cells of one level-8 cell

# one shard per chunk of the conversion, in inner chunks of one level-8 cell
_SHARDED_ZSTD_CODECS = [
    {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": [_L8_INNER_CHUNK],
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
            ],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"},
            ],
            "index_location": "end",
        },
    }
]

# ── Licence and attribution ───────────────────────────────────────────────────
# required by the licence of the data (written on the output STAC item, and as
# attributes of the root group by the example notebook)
DEM_GLO30_LICENSE_URL = (
    "https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/"
    "copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf"
)
DEM_GLO30_ROOT_ATTRS: dict[str, str] = {
    "title": "Copernicus DEM GLO-30 on the HEALPix grid",
    "source": "Copernicus DEM GLO-30 (COP-DEM_GLO-30-DGED), regridded to HEALPix",
    "copyright": (
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
        "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European "
        "Union and ESA; all rights reserved"
    ),
    "license": (
        "ESA User License for Copernicus Contributing Mission data (Copernicus DEM)"
    ),
    "license_url": DEM_GLO30_LICENSE_URL,
    "references": "https://doi.org/10.5270/ESA-c5d3d65",
    "source_url": (
        "https://dataspace.copernicus.eu/explore-data/data-collections/"
        "copernicus-contributing-missions/collections-description/COP-DEM"
    ),
}

# ── ConvertSettings dict ──────────────────────────────────────────────────────
DEM_GLO30_CONVERT_SETTINGS: dict = {
    "metadata": {
        "stac": {
            "override_properties": {"license": "other"},
            "override_links": [
                {
                    "rel": "license",
                    "href": DEM_GLO30_LICENSE_URL,
                    "type": "application/pdf",
                    "title": DEM_GLO30_ROOT_ATTRS["license"],
                },
            ],
        },
    },
    "multiscale_settings": {
        "groups": ["dem"],
        "pyramid_groups": {
            "dem": {"derived_levels": list(range(17, 7, -1)), "reduce": "mean"},
        },
    },
    "group_settings": {
        # ── DSM, 1 arc-second input – bilinear resampling at level 18 (~25 m) ─
        "dem/glo30": {
            "healpix": _HEALPIX_L18_WGS84,
            "chunk": {
                "method": "healpix_cell_dense",
                "healpix": _HEALPIX_L6_SHARDS,
                "chunk_buffer_width": 300.0,  # metres: a few input pixels
            },
            "resampler": {
                "name": "bilinear",
                # ring_weight 2 instead of 3: the same cells and values, with
                # about 30% less memory. batch_size (healpix-resample >=
                # 2026.10.2) bounds the memory of the neighbour search: a
                # level-6 shard (about 32 million input points) peaks near
                # 15 GiB instead of more than 115 GiB
                "init_params": {
                    "threshold": 0.05,
                    "ring_weight": 2,
                    "batch_size": 2_000_000,
                },
            },
            # the source heights are float32: float64 would add no precision
            "output_dtype": "float32",
            "codecs": _SHARDED_ZSTD_CODECS,
            # the cell ids of a shard are one contiguous range: with a delta
            # filter they take about 12 kB per shard instead of 17 MB
            "healpix_coordinate_codecs": [
                {"name": "numcodecs.delta", "configuration": {"dtype": "<u8"}},
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "zstd", "configuration": {"level": 0, "checksum": False}},
            ],
        },
    },
}
