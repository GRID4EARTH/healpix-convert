"""The codecs of the cell ids coordinate are a group setting (#32)."""

from __future__ import annotations

import numpy as np
import xarray as xr
import zarr

from healpix_convert import create_healpix_dataset
from healpix_convert.settings.common import ConvertSettings

STAC_ITEM = {
    "type": "Feature",
    "stac_version": "1.1.0",
    "stac_extensions": [],
    "id": "synthetic-input",
    "bbox": [-1.0, -1.0, 1.0, 1.0],
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]],
    },
    "properties": {"datetime": "2026-10-15T00:00:00Z"},
    "links": [],
    "assets": {},
}
HEALPIX = {"indexing_scheme": "nested", "ellipsoid": {"name": "wgs84"}}
DELTA = [
    {"name": "numcodecs.delta", "configuration": {"dtype": "<u8"}},
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "zstd", "configuration": {"level": 3, "checksum": False}},
]


def test_cell_ids_codecs(tmp_path):
    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.ones((32, 32), dtype="float32"))},
        coords={
            "lon": ("lon", np.linspace(-1.0, 1.0, 32), {"standard_name": "longitude"}),
            "lat": ("lat", np.linspace(-1.0, 1.0, 32), {"standard_name": "latitude"}),
        },
    )
    tree = xr.DataTree.from_dict({"/measurements": ds})
    tree.attrs["stac_discovery"] = STAC_ITEM
    input_path = str(tmp_path / "input.zarr")
    tree.to_zarr(input_path, mode="w", consolidated=True)
    settings = ConvertSettings.model_validate(
        {
            "group_settings": {
                "measurements": {
                    "healpix": {"refinement_level": 8, **HEALPIX},
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {"refinement_level": 5, **HEALPIX},
                    },
                    "resampler": {"name": "nearest"},
                    "healpix_coordinate_codecs": DELTA,
                }
            }
        }
    )
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    cell_ids = zarr.open_group(output_path, mode="r")["measurements/cell_ids"]
    actual = [codec["name"] for codec in cell_ids.metadata.to_dict()["codecs"]]
    assert actual == ["numcodecs.delta", "bytes", "zstd"]
    np.testing.assert_array_equal(np.diff(cell_ids[:].reshape(-1, 4**3), axis=1), 1)
