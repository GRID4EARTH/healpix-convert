"""The cell ids coordinate is delta-coded before zstd (#32)."""

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


def test_cell_ids_are_delta_coded(tmp_path):
    lon = np.linspace(-1.0, 1.0, 32)
    lat = np.linspace(-1.0, 1.0, 32)
    ds = xr.Dataset(
        {"t2m": (("lat", "lon"), np.ones((32, 32), dtype="float32"))},
        coords={
            "lon": (
                "lon",
                lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "lat": (
                "lat",
                lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
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
                    "healpix": {
                        "refinement_level": 8,
                        "indexing_scheme": "nested",
                        "ellipsoid": {"name": "wgs84"},
                    },
                    "chunk": {
                        "method": "healpix_cell_dense",
                        "healpix": {
                            "refinement_level": 5,
                            "indexing_scheme": "nested",
                            "ellipsoid": {"name": "wgs84"},
                        },
                    },
                    "resampler": {"name": "nearest"},
                }
            }
        }
    )
    output_path = str(tmp_path / "output.zarr")
    create_healpix_dataset([input_path], settings, output_path)

    cell_ids = zarr.open_group(output_path, mode="r")["measurements/cell_ids"]
    actual = [codec["name"] for codec in cell_ids.metadata.to_dict()["codecs"]]
    assert actual[0] == "numcodecs.delta"

    # dense chunks: each chunk holds one contiguous range of ids
    ids = cell_ids[:].reshape(-1, 4 ** (8 - 5))
    np.testing.assert_array_equal(np.diff(ids, axis=1), 1)
